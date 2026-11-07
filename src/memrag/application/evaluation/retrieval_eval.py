"""Exact (brute-force cosine) retrieval over gold facts, scored against the labels.

Vectors come in already L2-normalized, so the dot product is the cosine similarity.
The ranking metrics ignore no-answer queries; those only enter the top-1 score AUC.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from memrag.application.evaluation.dataset_check import GoldSet
from memrag.domain.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k
from memrag.domain.metrics.statistics import auc, bootstrap_ci, mean, paired_bootstrap_ci
from memrag.domain.schemas.evaluation import GoldFact, Split, TestQuery

# all: every user's memories; person: the asker's only (the robot always knows who is talking);
# person+filter: also the query's date/category filter
SCOPES = ("all", "person", "person+filter")
PRIMARY_SCOPE = "person+filter"
RECALL_KS = (1, 3, 5, 10)
TOP_N = 10


@dataclass
class EvalInput:
    queries: list[TestQuery]
    facts: list[GoldFact]
    grades: dict[str, dict[str, int]]  # query_id -> fact_id -> grade


def build_input(gs: GoldSet, split: Split | None) -> EvalInput:
    queries = [q for q in gs.queries if split is None or q.split == split]
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for label in gs.labels:
        grades[label.query_id][label.fact_id] = label.grade
    return EvalInput(queries, list(gs.facts), {q.query_id: dict(grades[q.query_id]) for q in queries})


def candidate_mask(facts: list[GoldFact], query: TestQuery, scope: str) -> np.ndarray:
    """Which facts the query may retrieve. Facts mentioned after the question are never visible."""
    f = query.filters
    keep = []
    for fact in facts:
        ok = fact.mention_date <= query.asked_on
        if scope != "all":
            ok = ok and fact.person_id == query.person_id
        if scope == "person+filter":
            if f.category is not None:
                ok = ok and fact.category == f.category
            if f.date_from or f.date_to:
                # A date filter excludes memories without an event date
                ok = ok and fact.event_date is not None
                ok = ok and (f.date_from is None or fact.event_date >= f.date_from)
                ok = ok and (f.date_to is None or fact.event_date <= f.date_to)
        keep.append(ok)
    return np.asarray(keep, dtype=bool)


def rank(query_vecs: np.ndarray, doc_vecs: np.ndarray, data: EvalInput) -> list[dict]:
    """One row per (query, scope) with the top results and the per-query metrics."""
    if query_vecs.shape[0] != len(data.queries) or doc_vecs.shape[0] != len(data.facts):
        raise ValueError("vector counts do not match the queries and facts")
    fact_ids = [f.fact_id for f in data.facts]
    owners = [f.person_id for f in data.facts]
    rows = []
    for qi, query in enumerate(data.queries):
        scores = doc_vecs @ query_vecs[qi]
        grades = data.grades[query.query_id]
        for scope in SCOPES:
            idx = np.flatnonzero(candidate_mask(data.facts, query, scope))
            order = idx[np.argsort(-scores[idx], kind="stable")]
            ranked = [fact_ids[i] for i in order]
            row = {
                "query_id": query.query_id,
                "query_type": query.query_type.value,
                "scope": scope,
                "candidates": len(idx),
                "answerable": bool(grades),
                "top": [[fact_ids[i], round(float(scores[i]), 4), grades.get(fact_ids[i], 0)] for i in order[:TOP_N]],
                "top1_score": round(float(scores[order[0]]), 4) if len(order) else None,
                "top1_other_person": bool(len(order)) and owners[order[0]] != query.person_id,
            }
            if grades:
                row.update({f"recall@{k}": recall_at_k(ranked, grades, k) for k in RECALL_KS})
                row["mrr@10"] = mrr_at_k(ranked, grades, 10)
                row["ndcg@10"] = ndcg_at_k(ranked, grades, 10)
            rows.append(row)
    return rows


def _with_ci(values: list[float]) -> dict:
    low, high = bootstrap_ci(values)
    return {"mean": round(mean(values), 4), "ci95": [round(low, 4), round(high, 4)]}


def summarize(rows: list[dict]) -> dict:
    """Per-scope means with bootstrap intervals, nDCG@10 by query type, and the no-answer AUC."""
    out: dict = {}
    for scope in SCOPES:
        scoped = [r for r in rows if r["scope"] == scope]
        ans = [r for r in scoped if r["answerable"]]
        s: dict = {"queries": len(ans)}
        for metric in ("ndcg@10", "mrr@10", *(f"recall@{k}" for k in RECALL_KS)):
            s[metric] = _with_ci([r[metric] for r in ans])
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in ans:
            by_type[r["query_type"]].append(r["ndcg@10"])
        s["ndcg@10_by_type"] = {t: round(mean(v), 4) for t, v in sorted(by_type.items())}
        s["top1_other_person_rate"] = round(mean([float(r["top1_other_person"]) for r in ans]), 4)
        no_ans = [r["top1_score"] for r in scoped if not r["answerable"] and r["top1_score"] is not None]
        if no_ans and ans:
            s["no_answer_auc"] = round(auc([r["top1_score"] for r in ans], no_ans), 4)
            s["no_answer_queries"] = len(no_ans)
        out[scope] = s
    return out


def compare(per_model: dict[str, list[dict]], metric: str = "ndcg@10") -> dict:
    """For each scope, every model's paired difference against the best model, with a 95% interval.

    An interval that contains 0 means the gap is not significant.
    """
    out: dict = {}
    for scope in SCOPES:
        values = {
            m: {r["query_id"]: r[metric] for r in rows if r["scope"] == scope and r["answerable"]}
            for m, rows in per_model.items()
        }
        best = max(values, key=lambda m: mean(list(values[m].values())))
        qids = sorted(values[best])
        table = {}
        for m, v in values.items():
            low, high = paired_bootstrap_ci([v[q] for q in qids], [values[best][q] for q in qids])
            diff = mean([v[q] for q in qids]) - mean([values[best][q] for q in qids])
            table[m] = {"diff": round(diff, 4), "ci95": [round(low, 4), round(high, 4)],
                        "significant": m != best and high < 0}
        out[scope] = {"best": best, "metric": metric, "vs_best": table}
    return out
