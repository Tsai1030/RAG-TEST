"""Plan 5.1.6 acceptance over the query set: refusals, citations and updated facts.

Answer quality is not scored this round. This checks that the pipeline answers when a
fully relevant (grade 2) memory exists, refuses when no memory is relevant, and cites
only memories that were in its context and inside the filter. Queries whose best memory
is only partly relevant (grade 1) are reported but not scored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from memrag.application.answer import Answer
from memrag.domain.metrics.statistics import bootstrap_ci, mean
from memrag.domain.schemas.evaluation import TestQuery
from memrag.domain.schemas.memory import MemoryRecord


def expected(grades: Mapping[str, int]) -> str:
    if any(g == 2 for g in grades.values()):
        return "answer"
    return "partial" if grades else "refuse"


def score_answer(query: TestQuery, answer: Answer, grades: Mapping[str, int], records: Mapping[str, MemoryRecord],
                 update: tuple[set[str], set[str]] | None = None) -> dict:
    """One row per query. `update` is (memories of the new fact, memories of the fact it replaced)."""
    cited = [grades.get(c, 0) for c in answer.citations]
    row = {
        "query_id": query.query_id, "query_type": query.query_type.value, "split": query.split.value,
        "expected": expected(grades), "refused": answer.refused, "reason": answer.reason,
        "citations": answer.citations, "cited_grades": cited, "invalid_citations": answer.invalid_citations,
        "outside_filter": [c for c in answer.citations if not answer.filter.matches(records[c])],
        "retrieved": answer.retrieved, "text": answer.text,
    }
    if update:
        new, old = update
        row["update"] = {"cited_new": bool(new & set(answer.citations)),
                         "cited_old": bool(old & set(answer.citations))}
    return row


def _rate(values: list[bool]) -> dict:
    if not values:
        return {"n": 0}
    low, high = bootstrap_ci([float(v) for v in values])
    return {"n": len(values), "rate": round(mean([float(v) for v in values]), 4), "ci95": [round(low, 4), round(high, 4)]}


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected"] == "answer"]
    ref = [r for r in rows if r["expected"] == "refuse"]
    part = [r for r in rows if r["expected"] == "partial"]
    answered = [r for r in ans if not r["refused"]]
    updates = [r for r in rows if "update" in r and not r["refused"]]
    return {
        "queries": len(rows),
        "answerable": {
            "answered": _rate([not r["refused"] for r in ans]),
            "cited_relevant": _rate([any(g >= 1 for g in r["cited_grades"]) for r in answered]),
            "cited_best": _rate([2 in r["cited_grades"] for r in answered]),
            "refused_by": dict(Counter(r["reason"] for r in ans if r["refused"])),
        },
        "no_answer": {
            "refused": _rate([r["refused"] for r in ref]),
            "refused_by": dict(Counter(r["reason"] for r in ref if r["refused"])),
        },
        "partial": {"answered": _rate([not r["refused"] for r in part])},
        "citations": {
            "invalid": sum(len(r["invalid_citations"]) for r in rows),
            "outside_filter": sum(len(r["outside_filter"]) for r in rows),
            "answered_without_citation": sum(not r["refused"] and not r["citations"] for r in rows),
        },
        "updates": {
            "queries": sum("update" in r for r in rows),
            "answered": len(updates),
            "cited_new": sum(r["update"]["cited_new"] for r in updates),
            "cited_old_only": sum(r["update"]["cited_old"] and not r["update"]["cited_new"] for r in updates),
        },
        "reasons": dict(Counter(r["reason"] for r in rows)),
    }
