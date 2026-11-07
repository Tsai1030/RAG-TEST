"""Ranking metrics for one query. `grades` maps memory ID to graded relevance (1 or 2); missing IDs are 0."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Relevant memories (grade >= 1) in the top k, divided by all relevant memories."""
    relevant = {i for i, g in grades.items() if g > 0}
    if not relevant:
        raise ValueError("recall is undefined for a query without relevant memories")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """1 / rank of the first relevant memory in the top k, or 0."""
    for rank, mem_id in enumerate(ranked[:k], start=1):
        if grades.get(mem_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """nDCG with linear gain (gain = grade) and log2(rank + 1) discount."""
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    if not ideal:
        raise ValueError("nDCG is undefined for a query without relevant memories")
    dcg = sum(grades.get(m, 0) / math.log2(r + 1) for r, m in enumerate(ranked[:k], start=1))
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg
