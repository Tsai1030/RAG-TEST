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


def _hits(store, vecs):
    return [[h.memory_id for h in store.search(v, 5, MemoryFilter(person_id="p01"))] for v in vecs[:20]]


def test_persists_and_reopens_with_the_same_results(tmp_path):
    records, vecs = _corpus()
    store = NumpyStore(vecs.shape[1], tmp_path, "ruri-v3-130m")
    store.upsert(records[:200], vecs[:200])
    store.upsert(records[200:], vecs[200:])  # appended
    folder = tmp_path / "mem__ruri-v3-130m__16"
    files = {p.name for p in folder.iterdir()}
    assert files == {"meta.json", "vectors.f32", "records.jsonl"}
    again = NumpyStore(vecs.shape[1], tmp_path, "ruri-v3-130m")
    assert again.count() == len(records) and _hits(again, vecs) == _hits(store, vecs)
    assert again.get([records[7].memory_id])[0] == records[7]


def test_rerun_writes_nothing_and_a_change_rewrites(tmp_path):
    records, vecs = _corpus()
    store = NumpyStore(vecs.shape[1], tmp_path, "m")
    store.upsert(records, vecs)
    vec_file = tmp_path / "mem__m__16" / "vectors.f32"
    before = (vec_file.stat().st_size, vec_file.stat().st_mtime_ns)
    store.upsert(records[:50], vecs[:50])  # same content: no write
    assert (vec_file.stat().st_size, vec_file.stat().st_mtime_ns) == before
    changed = records[3].model_copy(update={"content": "佐藤さんは別の記憶になった。"})
    store.upsert([changed], vecs[4:5])
    again = NumpyStore(vecs.shape[1], tmp_path, "m")
    assert again.count() == len(records)
    assert again.get([changed.memory_id])[0].content == "佐藤さんは別の記憶になった。"
    assert again.search(vecs[4], 1, MemoryFilter(person_id=changed.person_id))[0].score == pytest.approx(1.0)


def test_reopen_drops_a_half_written_row(tmp_path):
    records, vecs = _corpus(n=20)
    NumpyStore(vecs.shape[1], tmp_path, "m").upsert(records, vecs)
    folder = tmp_path / "mem__m__16"
    # cut off after the vector append but before the record line
    with (folder / "vectors.f32").open("ab") as f:
        f.write(np.ones(16, dtype=np.float32).tobytes())
    again = NumpyStore(vecs.shape[1], tmp_path, "m")
    assert again.count() == 20
    assert (folder / "vectors.f32").stat().st_size == 20 * 16 * 4  # repaired on disk too


def test_refuses_another_model_or_dimension(tmp_path):
    NumpyStore(16, tmp_path, "m").upsert(*_corpus(n=5))
    (tmp_path / "mem__m__16" / "meta.json").write_text('{"model": "other", "dim": 16, "format": 1}', encoding="utf-8")
    with pytest.raises(ConfigError):
        NumpyStore(16, tmp_path, "m")
    with pytest.raises(ConfigError):
        NumpyStore(16, tmp_path)  # a persistent store needs the model name


def test_bootstrap_picks_the_configured_store(tmp_path):
    from memrag import bootstrap
    from memrag.infrastructure.vectorstore.chroma_store import ChromaStore

    assert isinstance(bootstrap.vector_store("m", 16, fake=True), NumpyStore)
    assert isinstance(bootstrap.vector_store("m", 16, path=tmp_path), NumpyStore)
    chroma = bootstrap.vector_store("m", 16, fake=True, kind="chroma")
    assert isinstance(chroma, ChromaStore)
    chroma.close()
    with pytest.raises(ConfigError):
        bootstrap.vector_store("m", 16, fake=True, kind="faiss")


def test_same_id_twice_in_one_batch_keeps_the_last():
    records, vecs = _corpus(n=3)
    store = NumpyStore(16)
    store.upsert([records[0], records[0]], vecs[:2])
    assert store.count() == 1
    assert store.search(vecs[1], 1)[0].score == pytest.approx(1.0)
