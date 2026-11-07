"""Threshold calibration and small-candidate evaluation (plan 5.1.8).

With only a few memories per user, top-k always returns something, so the ranking
metrics cannot show wrong retrievals. Here each query is paired with small sampled
candidate sets of its user's memories:

- positive set: one memory that answers the query (grade 2) plus grade-0 memories
- negative set: grade-0 memories only. Answerable queries also yield negative sets
  (their relevant memories left out), standing for "the answer is not stored";
  the dev set has too few no-answer queries on its own.

Grade-1 memories (partly relevant) are never sampled: letting one through is not a
false recall, and blocking one does not block the answer.

A memory passes when its cosine score is >= the model's threshold. The threshold is
the lowest one (rounded up to 4 decimals) that keeps the false-recall rate on
negative sets within the target. Scores are compared in float64.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from memrag.application.evaluation.retrieval_eval import EvalInput
from memrag.domain.metrics.statistics import cluster_bootstrap_rate

SIZES = tuple(range(2, 11))
SAMPLES_PER_SIZE = 10
TARGET_FALSE_RECALL = 0.05
SEED = 0


@dataclass(frozen=True)
class CandidateSet:
    query_index: int
    doc_indices: tuple[int, ...]
    positive: bool  # holds exactly one relevant memory

    @property
    def size(self) -> int:
        return len(self.doc_indices)


def candidate_sets(data: EvalInput, sizes=SIZES, samples: int = SAMPLES_PER_SIZE, seed: int = SEED) -> list[CandidateSet]:
    """Deterministic sampled sets; the same data and seed always give the same sets."""
    by_person: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(data.docs):
        by_person[d.person_id].append(i)
    lost = set(data.unanswerable_in_corpus)
    sets = []
    for qi, q in enumerate(data.queries):
        if q.query_id in lost:
            continue
        rng = random.Random(f"{seed}:{q.query_id}")
        grades = data.grades[q.query_id]
        own = [i for i in by_person[q.person_id] if data.docs[i].visible_from <= q.asked_on]
        relevant = [i for i in own if grades.get(data.docs[i].doc_id, 0) == 2]
        irrelevant = [i for i in own if grades.get(data.docs[i].doc_id, 0) == 0]
        for n in sizes:
            for _ in range(samples):
                if relevant and len(irrelevant) >= n - 1:
                    docs = [rng.choice(relevant), *rng.sample(irrelevant, n - 1)]
                    sets.append(CandidateSet(qi, tuple(docs), True))
                if len(irrelevant) >= n:
                    sets.append(CandidateSet(qi, tuple(rng.sample(irrelevant, n)), False))
    return sets


def _scores(query_vecs: np.ndarray, doc_vecs: np.ndarray, s: CandidateSet) -> np.ndarray:
    # float64: a float32 score equal to the threshold must not pass because of rounding
    return doc_vecs[list(s.doc_indices)].astype(np.float64) @ query_vecs[s.query_index].astype(np.float64)


def calibrate(query_vecs: np.ndarray, doc_vecs: np.ndarray, sets: list[CandidateSet],
              target: float = TARGET_FALSE_RECALL) -> float:
    """Lowest 4-decimal threshold whose false-recall rate on the negative sets is <= target."""
    tops = sorted((float(_scores(query_vecs, doc_vecs, s).max()) for s in sets if not s.positive), reverse=True)
    if not tops:
        raise ValueError("no negative candidate sets to calibrate on")
    allowed = int(target * len(tops))  # this many negative sets may still pass
    if allowed >= len(tops):
        return -1.0
    t = math.ceil(tops[allowed] * 10_000) / 10_000
    return t if t > tops[allowed] else t + 0.0001


def evaluate(query_vecs: np.ndarray, doc_vecs: np.ndarray, data: EvalInput, sets: list[CandidateSet],
             threshold: float) -> dict:
    """Plan 5.1.8 threshold metrics, overall and per candidate-set size, with query-clustered 95% intervals."""
    def rel(s: CandidateSet, j: int) -> bool:
        q = data.queries[s.query_index]
        return data.grades[q.query_id].get(data.docs[s.doc_indices[j]].doc_id, 0) == 2

    per_size: dict = {}
    for n in ("all", *sorted({s.size for s in sets})):
        chosen = [s for s in sets if n == "all" or s.size == n]
        false_recall, false_block, top1_wrong = defaultdict(list), defaultdict(list), defaultdict(list)
        passed = wrong_passed = 0
        for s in chosen:
            sc = _scores(query_vecs, doc_vecs, s)
            ok = sc >= threshold
            qid = data.queries[s.query_index].query_id
            if s.positive:
                false_block[qid].append(not any(ok[j] and rel(s, j) for j in range(s.size)))
                top1_wrong[qid].append(not rel(s, int(sc.argmax())))
            else:
                false_recall[qid].append(bool(ok.any()))
            passed += int(ok.sum())
            wrong_passed += sum(1 for j in range(s.size) if ok[j] and not rel(s, j))
        per_size[str(n)] = {
            "positive_sets": sum(len(v) for v in false_block.values()),
            "negative_sets": sum(len(v) for v in false_recall.values()),
            "false_recall": _rate(false_recall),
            "false_block": _rate(false_block),
            "top1_wrong": _rate(top1_wrong),
            "wrong_pass_rate": round(wrong_passed / passed, 4) if passed else None,
        }
    return {"threshold": threshold, "by_size": per_size}


def _rate(groups: dict[str, list[bool]]) -> dict | None:
    if not groups:
        return None
    flat = [x for v in groups.values() for x in v]
    low, high = cluster_bootstrap_rate(list(groups.values()))
    return {"rate": round(sum(flat) / len(flat), 4), "ci95": [round(low, 4), round(high, 4)]}
