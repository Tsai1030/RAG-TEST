"""Plan 5.1.5 acceptance: an approximate VectorStore against the exact reference.

- filter translation: for each query's filters and for random single/combined filters,
  both stores must accept exactly the same memories
- person isolation: no hit may belong to another user
- HNSW recall@k: overlap of the approximate top-k with the exact top-k
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from statistics import mean
from typing import Sequence

from memrag.domain.ports import VectorStore
from memrag.domain.schemas.evaluation import TestQuery
from memrag.domain.schemas.memory import Category, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter

RANDOM_FILTERS = 300


def query_filters(q: TestQuery) -> dict[str, MemoryFilter]:
    f = q.filters
    return {
        "all": MemoryFilter(),
        "person": MemoryFilter(person_id=q.person_id),
        "person+filter": MemoryFilter(person_id=q.person_id, category=f.category, date_from=f.date_from,
                                      date_to=f.date_to),
    }


def random_filters(records: Sequence[MemoryRecord], n: int = RANDOM_FILTERS, seed: int = 0) -> list[MemoryFilter]:
    """Single and combined filters over persons, categories and date ranges (including open-ended ones)."""
    rng = random.Random(seed)
    persons = sorted({r.person_id for r in records})
    out = []
    for _ in range(n):
        kw: dict = {}
        if rng.random() < 0.6:
            kw["person_id"] = rng.choice(persons + ["p_unknown"])
        if rng.random() < 0.4:
            kw["category"] = rng.choice(list(Category))
        if rng.random() < 0.4:
            start = date(2026, 9, 25) + timedelta(days=rng.randrange(80))
            end = start + timedelta(days=rng.randrange(1, 40))
            choice = rng.randrange(3)
            if choice in (0, 2):
                kw["date_from"] = start
            if choice in (1, 2):
                kw["date_to"] = end
        out.append(MemoryFilter(**kw))
    return out


def compare(records: Sequence[MemoryRecord], queries: Sequence[TestQuery], query_vecs, exact: VectorStore,
            approx: VectorStore, k: int = 10) -> dict:
    owner = {r.memory_id: r.person_id for r in records}
    filter_mismatches = []
    for flt in random_filters(records):
        if exact.matching_ids(flt) != approx.matching_ids(flt):
            filter_mismatches.append(flt.model_dump(mode="json", exclude_none=True))
    by_scope: dict = {}
    for scope in ("all", "person", "person+filter"):
        recalls, cross, short = [], 0, 0
        for q, vec in zip(queries, query_vecs):
            flt = query_filters(q)[scope]
            if scope != "all" and exact.matching_ids(flt) != approx.matching_ids(flt):
                filter_mismatches.append({"query_id": q.query_id, "scope": scope})
            truth = [h.memory_id for h in exact.search(vec, k, flt)]
            got = [h.memory_id for h in approx.search(vec, k, flt)]
            if scope != "all":
                cross += sum(owner[i] != q.person_id for i in got)
            short += len(got) < len(truth)
            if truth:
                recalls.append(len(set(truth) & set(got)) / len(truth))
        by_scope[scope] = {"queries": len(queries), f"recall@{k}": round(mean(recalls), 4) if recalls else None,
                           "queries_below_full_recall": sum(r < 1 for r in recalls),
                           "cross_user_hits": cross, "queries_with_fewer_hits": short}
    return {"random_filters": RANDOM_FILTERS, "filter_mismatches": filter_mismatches, "scopes": by_scope}
