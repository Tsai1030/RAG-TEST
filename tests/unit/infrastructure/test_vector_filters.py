from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter
from memrag.infrastructure.vectorstore.chroma_store import collection_name, to_where
from memrag.infrastructure.vectorstore.exact_store import ExactStore


def _rec(person, content, category="event", event_date=None):
    d = MemoryDraft(category=category, content=content, event_date=event_date, source_turn_ids=[1])
    return MemoryRecord.from_draft(d, person_id=person, conversation_date=date(2026, 10, 5), source_log_id="d")


RECS = [_rec("p01", "佐藤さんは散歩した。", event_date=date(2026, 10, 3)),
        _rec("p01", "佐藤さんはバラが好き。", category="preference"),
        _rec("p02", "鈴木さんは将棋をした。", event_date=date(2026, 10, 20))]


def test_filter_reference_semantics():
    assert [MemoryFilter(person_id="p01").matches(r) for r in RECS] == [True, True, False]
    assert [MemoryFilter(category="preference").matches(r) for r in RECS] == [False, True, False]
    # A date bound excludes memories without an event date, even an open-ended one
    assert [MemoryFilter(date_to=date(2026, 12, 31)).matches(r) for r in RECS] == [True, False, True]
    assert [MemoryFilter(date_from=date(2026, 10, 10)).matches(r) for r in RECS] == [False, False, True]
    with pytest.raises(ValidationError):
        MemoryFilter(date_from=date(2026, 11, 2), date_to=date(2026, 11, 1))


def test_date_fallback_matches_undated_memories_by_conversation_date():
    # RECS were told on 2026-10-05; only the undated one falls back to that day
    flt = MemoryFilter(date_from=date(2026, 10, 4), date_to=date(2026, 10, 6), date_fallback=True)
    assert [flt.matches(r) for r in RECS] == [False, True, False]
    assert [MemoryFilter(date_to=date(2026, 10, 4), date_fallback=True).matches(r) for r in RECS] == [True, False, False]
    # Without a date bound the fallback changes nothing
    assert MemoryFilter(date_fallback=True).matches(RECS[1])


def test_to_where():
    assert to_where(None) is None and to_where(MemoryFilter()) is None
    assert to_where(MemoryFilter(person_id="p01")) == {"person_id": {"$eq": "p01"}}
    w = to_where(MemoryFilter(person_id="p01", category="plan", date_to=date(2026, 11, 30)))
    assert w == {"$and": [{"person_id": {"$eq": "p01"}}, {"category": {"$eq": "plan"}},
                          {"event_date_ord": {"$gt": 0}}, {"event_date_ord": {"$lte": 20261130}}]}
    w = to_where(MemoryFilter(date_from=date(2026, 10, 1), date_fallback=True))
    assert w == {"$or": [{"$and": [{"event_date_ord": {"$gt": 0}}, {"event_date_ord": {"$gte": 20261001}}]},
                         {"$and": [{"event_date_ord": {"$eq": 0}}, {"conversation_date_ord": {"$gte": 20261001}}]}]}


def test_collection_name_is_safe_and_per_dim():
    assert collection_name("bge-base-zh-v1.5", 768) == "mem__bge-base-zh-v1_5__768"
    assert collection_name("x", 256) != collection_name("x", 512)


def test_exact_store():
    store = ExactStore(2)
    vecs = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    store.upsert(RECS, vecs)
    store.upsert(RECS[:1], vecs[:1])  # idempotent
    assert store.count() == 3
    hits = store.search([1.0, 0.0], 5, MemoryFilter(person_id="p01"))
    assert [h.memory_id for h in hits] == [RECS[0].memory_id, RECS[1].memory_id]
    assert hits[0].score == pytest.approx(1.0)
    assert store.search([1.0, 0.0], 5, MemoryFilter(person_id="nobody")) == []
    with pytest.raises(Exception):
        store.upsert(RECS[:1], np.ones((1, 3)))
