"""Exact (brute-force cosine) retrieval over a memory corpus, scored against the labels.

The corpus is either the gold facts themselves (ideal summaries) or the memories
summarized from the dialogue logs. A memory inherits the best grade of the gold
facts it was summarized from. Vectors come in already L2-normalized, so the dot
product is the cosine similarity. The ranking metrics ignore no-answer queries;
those only enter the top-1 score AUC.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from memrag.application.evaluation.dataset_check import GoldSet
from memrag.domain.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k
from memrag.domain.metrics.statistics import auc, bootstrap_ci, mean, paired_bootstrap_ci
from memrag.domain.schemas.evaluation import MemoryLabel, Split, TestQuery
from memrag.domain.schemas.memory import Category, MemoryRecord

# all: every user's memories; person: the asker's only (the robot always knows who is talking);
# person+filter: also the query's date/category filter
SCOPES = ("all", "person", "person+filter")
PRIMARY_SCOPE = "person+filter"
RECALL_KS = (1, 3, 5, 10)
TOP_N = 10


@dataclass(frozen=True)
class Doc:
    doc_id: str
    person_id: str
    text: str
    category: Category
    event_date: date | None
    visible_from: date  # the day the robot heard it


@dataclass
class EvalInput:
    queries: list[TestQuery]
    docs: list[Doc]
    grades: dict[str, dict[str, int]]  # query_id -> doc_id -> grade
    unanswerable_in_corpus: list[str]  # labelled answerable, but no doc carries a relevant fact


def _fact_grades(gs: GoldSet, queries: list[TestQuery]) -> dict[str, dict[str, int]]:
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for label in gs.labels:
        grades[label.query_id][label.fact_id] = label.grade
    return {q.query_id: dict(grades[q.query_id]) for q in queries}


def build_input(gs: GoldSet, split: Split | None, memories: list[MemoryRecord] | None = None,
                memory_labels: list[MemoryLabel] | None = None) -> EvalInput:
    """Queries of the split against the gold facts, or against `memories` when given.

    A memory's grade is the best grade of its gold facts, unless a pooled judgement
    in `memory_labels` says otherwise.
    """
    queries = [q for q in gs.queries if split is None or q.split == split]
    fact_grades = _fact_grades(gs, queries)
    if memories is None:
        docs = [Doc(f.fact_id, f.person_id, f.statement, f.category, f.event_date, f.mention_date) for f in gs.facts]
        return EvalInput(queries, docs, fact_grades, [])
    docs = [Doc(m.memory_id, m.person_id, m.content, m.category, m.event_date, m.conversation_date) for m in memories]
    judged = {(lab.query_id, lab.memory_id): lab.grade for lab in memory_labels or ()}
    grades = {}
    for q in queries:
        g = {}
        for m in memories:
            best = max((fact_grades[q.query_id].get(f, 0) for f in m.gold_fact_ids), default=0)
            best = judged.get((q.query_id, m.memory_id), best)
            if best:
                g[m.memory_id] = best
        grades[q.query_id] = g
    lost = [q.query_id for q in queries if fact_grades[q.query_id] and not grades[q.query_id]]
    return EvalInput(queries, docs, grades, lost)


def candidate_mask(docs: list[Doc], query: TestQuery, scope: str) -> np.ndarray:
    """Which docs the query may retrieve. Docs heard after the question are never visible."""
    f = query.filters
    keep = []
    for doc in docs:
        ok = doc.visible_from <= query.asked_on
        if scope != "all":
            ok = ok and doc.person_id == query.person_id
        if scope == "person+filter":
            if f.category is not None:
                ok = ok and doc.category == f.category
            if f.date_from or f.date_to:
                # A date filter excludes memories without an event date
                ok = ok and doc.event_date is not None
                ok = ok and (f.date_from is None or doc.event_date >= f.date_from)
                ok = ok and (f.date_to is None or doc.event_date <= f.date_to)
        keep.append(ok)
    return np.asarray(keep, dtype=bool)


def rank(query_vecs: np.ndarray, doc_vecs: np.ndarray, data: EvalInput) -> list[dict]:
    """One row per (query, scope) with the top results and the per-query metrics.

    Queries whose relevant facts never made it into the corpus are skipped.
    """
    if query_vecs.shape[0] != len(data.queries) or doc_vecs.shape[0] != len(data.docs):
        raise ValueError("vector counts do not match the queries and docs")
    doc_ids = [d.doc_id for d in data.docs]
    owners = [d.person_id for d in data.docs]
    lost = set(data.unanswerable_in_corpus)
    rows = []
    for qi, query in enumerate(data.queries):
        if query.query_id in lost:
            continue
        scores = doc_vecs @ query_vecs[qi]
        grades = data.grades[query.query_id]
        for scope in SCOPES:
            idx = np.flatnonzero(candidate_mask(data.docs, query, scope))
            order = idx[np.argsort(-scores[idx], kind="stable")]
            ranked = [doc_ids[i] for i in order]
            row = {
                "query_id": query.query_id,
                "query_type": query.query_type.value,
                "scope": scope,
                "candidates": len(idx),
                "answerable": bool(grades),
                "top": [[doc_ids[i], round(float(scores[i]), 4), grades.get(doc_ids[i], 0)] for i in order[:TOP_N]],
                "top1_score": round(float(scores[order[0]]), 4) if len(order) else None,
                "top1_other_person": bool(len(order)) and owners[order[0]] != query.person_id,
            }
            if grades and any(grades.get(d, 0) for d in ranked):
                row.update({f"recall@{k}": recall_at_k(ranked, grades, k) for k in RECALL_KS})
                row["mrr@10"] = mrr_at_k(ranked, grades, 10)
                row["ndcg@10"] = ndcg_at_k(ranked, grades, 10)
            elif grades:
                # Every relevant memory was filtered out (e.g. the LLM gave it another category)
                row.update({f"recall@{k}": 0.0 for k in RECALL_KS})
                row["mrr@10"] = 0.0
                row["ndcg@10"] = 0.0
                row["relevant_filtered_out"] = True
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
        s: dict = {"queries": len(ans), "relevant_filtered_out": sum(r.get("relevant_filtered_out", False) for r in ans)}
        if not ans:
            out[scope] = s
            continue
        for metric in ("ndcg@10", "mrr@10", *(f"recall@{k}" for k in RECALL_KS)):
            s[metric] = _with_ci([r[metric] for r in ans])
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in ans:
            by_type[r["query_type"]].append(r["ndcg@10"])
        s["ndcg@10_by_type"] = {t: round(mean(v), 4) for t, v in sorted(by_type.items())}
        s["top1_other_person_rate"] = round(mean([float(r["top1_other_person"]) for r in ans]), 4)
        no_ans = [r["top1_score"] for r in scoped if not r["answerable"] and r["top1_score"] is not None]
        if no_ans and ans:
            s["no_answer_auc"] = round(auc([r["top1_score"] for r in ans if r["top1_score"] is not None], no_ans), 4)
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
        if not any(values.values()):
            continue
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
