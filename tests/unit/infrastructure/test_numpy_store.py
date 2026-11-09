"""NumpyStore must give the same results as the ExactStore reference, only faster."""

import random
from datetime import date, timedelta

import numpy as np
import pytest

from memrag.application.evaluation.store_check import random_filters
from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import Category, MemoryDraft, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter
from memrag.infrastructure.vectorstore.exact_store import ExactStore
from memrag.infrastructure.vectorstore.numpy_store import NumpyStore


def _corpus(n=300, dim=16, seed=0):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        told = date(2026, 9, 25) + timedelta(days=rng.randrange(80))
        event = told + timedelta(days=rng.randrange(-10, 20)) if rng.random() < 0.3 else None
        d = MemoryDraft(category=rng.choice(list(Category)), content=f"佐藤さんの記憶その{i}です。",
                        event_date=event, source_turn_ids=[1])
        records.append(MemoryRecord.from_draft(d, person_id=f"p{rng.randrange(5):02d}", conversation_date=told,
                                               source_log_id=f"d{i}"))
    # not `seed` itself: expand_corpus draws its noise from default_rng(0)
    vecs = np.random.default_rng(seed + 1000).normal(size=(n, dim)).astype(np.float32)
    return records, vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _stores():
    records, vecs = _corpus()
    exact, fast = ExactStore(vecs.shape[1]), NumpyStore(vecs.shape[1])
    exact.upsert(records, vecs)
    fast.upsert(records, vecs)
    return records, vecs, exact, fast


def test_filters_match_the_reference():
    records, _, exact, fast = _stores()
    rng = random.Random(1)
    flts = random_filters(records, n=200) + [
        MemoryFilter(person_id="p01", categories=tuple(rng.sample(list(Category), 3))) for _ in range(20)]
    for flt in flts:
        assert fast.matching_ids(flt) == exact.matching_ids(flt), flt
        assert fast.match_count(flt) == len(exact.matching_ids(flt))


def test_search_matches_the_reference():
    _, vecs, exact, fast = _stores()
    rng = np.random.default_rng(2)
    for flt in [None, MemoryFilter(person_id="p02"),
                MemoryFilter(person_id="p03", date_from=date(2026, 10, 10), date_fallback=True)]:
        for _ in range(10):
            q = rng.normal(size=vecs.shape[1]).astype(np.float32)
            for k in (1, 3, 5, 500):
                a, b = exact.search(q, k, flt), fast.search(q, k, flt)
                assert [h.memory_id for h in a] == [h.memory_id for h in b]
                assert [h.score for h in a] == pytest.approx([h.score for h in b], abs=1e-5)


def test_upsert_get_and_edge_cases():
    records, vecs, _, fast = _stores()
    fast.upsert(records[:10], vecs[:10])  # idempotent
    assert fast.count() == len(records)
    ids = [records[5].memory_id, "nope", records[2].memory_id]
    assert [r.memory_id for r in fast.get(ids)] == [records[5].memory_id, records[2].memory_id]
    assert fast.get([records[5].memory_id])[0] == records[5]
    assert fast.search(vecs[0], 3, MemoryFilter(person_id="nobody")) == []
    assert fast.search(vecs[0], 0) == []
    assert fast.nbytes() > vecs.nbytes
    with pytest.raises(ConfigError):
        fast.upsert(records[:1], np.ones((1, 3)))
