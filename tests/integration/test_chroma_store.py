"""ChromaStore matches the exact store on random filters, never crosses users, and guards its collection."""

import random
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from memrag.application.evaluation.store_check import random_filters
from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import Category, MemoryDraft, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter
from memrag.infrastructure.vectorstore.chroma_store import ChromaStore
from memrag.infrastructure.vectorstore.exact_store import ExactStore


def _records(n=120, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        dated = rng.random() < 0.6
        d = MemoryDraft(category=rng.choice(list(Category)), content=f"記憶その{i}です。",
                        event_date=date(2026, 10, 1) + timedelta(days=rng.randrange(60)) if dated else None,
                        source_turn_ids=[1])
        out.append(MemoryRecord.from_draft(d, person_id=f"p{rng.randrange(5):02d}",
                                           conversation_date=date(2026, 10, 1) + timedelta(days=rng.randrange(40)),
                                           source_log_id=f"d{i}"))
    return out


@pytest.fixture()
def stores():
    recs = _records()
    vecs = np.random.default_rng(0).standard_normal((len(recs), 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    folder = Path(tempfile.mkdtemp())
    chroma = ChromaStore("test-model", 16, folder)
    chroma.upsert(recs, vecs)
    exact = ExactStore(16)
    exact.upsert(recs, vecs)
    yield recs, vecs, exact, chroma, folder
    chroma.close()


def test_filters_match_exact_store(stores):
    recs, vecs, exact, chroma, _ = stores
    for flt in random_filters(recs, n=200, seed=3):
        assert exact.matching_ids(flt) == chroma.matching_ids(flt), flt


def test_search_never_crosses_users_and_matches_exact_top_k(stores):
    recs, vecs, exact, chroma, _ = stores
    rng = np.random.default_rng(5)
    for _ in range(30):
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        person = f"p{rng.integers(5):02d}"
        flt = MemoryFilter(person_id=person)
        got = chroma.search(q, 5, flt)
        owners = {r.memory_id: r.person_id for r in recs}
        assert all(owners[h.memory_id] == person for h in got)
        assert [h.memory_id for h in got] == [h.memory_id for h in exact.search(q, 5, flt)]
        assert got[0].score == pytest.approx(exact.search(q, 1, flt)[0].score, abs=1e-4)


def test_upsert_is_idempotent_and_empty_results(stores):
    recs, vecs, exact, chroma, _ = stores
    chroma.upsert(recs[:10], vecs[:10])
    assert chroma.count() == len(recs)
    assert chroma.search(vecs[0], 5, MemoryFilter(person_id="p_none")) == []


def test_get_returns_full_records_in_order(stores):
    recs, vecs, exact, chroma, _ = stores
    ids = [recs[3].memory_id, "m_unknown", recs[0].memory_id]
    assert chroma.get(ids) == [recs[3], recs[0]] == exact.get(ids)
    assert chroma.get([]) == []


def test_collection_guards_model_and_dimension(stores):
    recs, vecs, exact, chroma, folder = stores
    with pytest.raises(ConfigError):
        chroma.upsert(recs[:1], np.ones((1, 8)))  # wrong dimension
    ChromaStore("a.b", 16, folder)  # collection mem__a_b__16
    with pytest.raises(ConfigError):
        ChromaStore("a_b", 16, folder)  # same collection name, another model
