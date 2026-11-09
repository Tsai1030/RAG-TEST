"""Plan 5.1.7 retrieval and scale test: test data, filter scenarios and summaries.

- `expand_corpus`: copy the frozen memories to N records for virtual users, with shifted
  dates and slightly perturbed vectors (identical vectors would distort HNSW)
- `scale_queries`: one filter per scenario for each query, from no filter to narrow ones
- `summarize_*`: latency percentiles, QPS, recall against labels and against exact search

Timings come from a child process per configuration (infrastructure.retrieval_probe).
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from memrag.domain.metrics.retrieval import recall_at_k
from memrag.domain.schemas.memory import Category, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter

USERS = 10
MEMORIES_PER_DAY = 5
EXPANDED_END = date(2026, 10, 31)
SCALE_SCENARIOS = ("none", "person", "person+week", "person+week+fallback", "person+month+fallback",
                   "person+category", "person+3categories", "category")


def noise_scale(dim: int, cosine: float) -> float:
    """Per-dimension Gaussian std whose noise leaves a unit vector at about `cosine` to the original."""
    return math.sqrt((1 / cosine**2 - 1) / dim)


def expand_corpus(records: Sequence[MemoryRecord], vectors, total: int, users: int = USERS,
                  noise_cosine: float = 0.98, seed: int = 0) -> tuple[list[MemoryRecord], np.ndarray]:
    """`total` memories for `users` virtual users (v00, v01, ...), drawn from `records`.

    Each user tells MEMORIES_PER_DAY memories a day, so the span grows with the corpus
    (5,000 per user is about 3 years); an event keeps its distance to the day it was told.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    per_user = total // users
    span = max(60, math.ceil(per_user / MEMORIES_PER_DAY))
    start = EXPANDED_END - timedelta(days=span - 1)
    out, picks = [], []
    for u in range(users):
        for j in range(per_user):
            i = rng.randrange(len(records))
            base = records[i]
            told = start + timedelta(days=rng.randrange(span))
            event = told + (base.event_date - base.conversation_date) if base.event_date else None
            out.append(base.model_copy(update={
                "person_id": f"v{u:02d}", "memory_id": f"x{u:02d}_{j:06d}", "source_log_id": f"x{u:02d}_{j:06d}",
                "conversation_date": told, "event_date": event,
                "event_date_ord": int(event.strftime("%Y%m%d")) if event else None, "gold_fact_ids": [],
            }))
            picks.append(i)
    noisy = vectors[picks] + np_rng.normal(0, noise_scale(vectors.shape[1], noise_cosine),
                                           (len(picks), vectors.shape[1])).astype(np.float32)
    noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
    return out, noisy


def user_spans(records: Sequence[MemoryRecord]) -> dict[str, tuple[date, date]]:
    spans: dict[str, tuple[date, date]] = {}
    for r in records:
        lo, hi = spans.get(r.person_id, (r.conversation_date, r.conversation_date))
        spans[r.person_id] = (min(lo, r.conversation_date), max(hi, r.conversation_date))
    return spans


def scale_queries(query_ids: Sequence[str], spans: Mapping[str, tuple[date, date]],
                  seed: int = 0) -> list[dict[str, MemoryFilter]]:
    """One filter per SCALE_SCENARIOS for each query; query i asks as user i mod len(spans)."""
    rng = random.Random(seed)
    users = sorted(spans)
    out = []
    for i, _ in enumerate(query_ids):
        user = users[i % len(users)]
        lo, hi = spans[user]
        anchor = lo + timedelta(days=rng.randrange((hi - lo).days + 1))
        cats = rng.sample(list(Category), 3)
        out.append({
            "none": MemoryFilter(),
            "person": MemoryFilter(person_id=user),
            "person+week": MemoryFilter(person_id=user, date_from=anchor - timedelta(days=6), date_to=anchor),
            "person+week+fallback": MemoryFilter(person_id=user, date_from=anchor - timedelta(days=6),
                                                 date_to=anchor, date_fallback=True),
            "person+month+fallback": MemoryFilter(person_id=user, date_from=anchor - timedelta(days=29),
                                                  date_to=anchor, date_fallback=True),
            "person+category": MemoryFilter(person_id=user, category=cats[0]),
            "person+3categories": MemoryFilter(person_id=user, categories=tuple(cats)),
            "category": MemoryFilter(category=cats[0]),
        })
    return out


def percentiles(samples: Sequence[float]) -> dict:
    xs = sorted(samples)
    if not xs:
        return {"n": 0}

    def at(p: float) -> float:
        return round(xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))], 3)

    return {"n": len(xs), "mean": round(mean(xs), 3), "p50": at(0.50), "p95": at(0.95), "p99": at(0.99),
            "max": round(xs[-1], 3)}


def summarize_latency(samples_ms: Mapping[str, Mapping[str, list]]) -> dict:
    """Per Top-k: percentiles of each stage and of the total, and sequential QPS."""
    out = {}
    for k, stages in samples_ms.items():
        embed = [x for x in stages["embed"] if x is not None]
        totals = [(e or 0.0) + s + g for e, s, g in zip(stages["embed"], stages["search"], stages["get"])]
        out[k] = {"embed": percentiles(embed) if embed else None, "search": percentiles(stages["search"]),
                  "get": percentiles(stages["get"]), "total": percentiles(totals),
                  "qps": round(1000 / mean(totals), 2) if totals else None}
    return out


def overlap_recall(approx: Mapping[str, Sequence[str]], exact: Mapping[str, Sequence[str]]) -> float | None:
    """Mean share of the exact top-k that the approximate search also returned."""
    vals = [len(set(approx.get(q, ())) & set(ids)) / len(ids) for q, ids in exact.items() if ids]
    return round(mean(vals), 4) if vals else None


def label_recall(hits: Mapping[str, Sequence[str]], grades: Mapping[str, Mapping[str, int]], k: int) -> dict:
    """Recall@k against the labels, and how often a grade-2 memory is in the top k (answerable queries only)."""
    rec, best = [], []
    for qid, ranked in hits.items():
        g = grades.get(qid) or {}
        if not any(v > 0 for v in g.values()):
            continue
        rec.append(recall_at_k(ranked, g, k))
        if any(v == 2 for v in g.values()):
            best.append(any(g.get(m) == 2 for m in ranked[:k]))
    return {"queries": len(rec), "recall": round(mean(rec), 4) if rec else None,
            "best_in_top_k": round(mean(best), 4) if best else None}


def selectivity(matched: Mapping[str, int], total: int) -> dict:
    """How much of the corpus a scenario's filters leave, over the queries."""
    vals = list(matched.values())
    return {"mean_matched": round(mean(vals), 1), "mean_share": round(mean(vals) / total, 5) if total else None,
            "min_matched": min(vals), "max_matched": max(vals)}
