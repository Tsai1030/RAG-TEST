"""Checks, statistics, split assignment and annotator agreement for the gold set."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from memrag.domain.metrics.statistics import cohen_kappa
from memrag.domain.schemas.evaluation import GoldFact, Persona, QueryLabel, QueryType, Split, TestQuery

# Target share of each query type (plan 5.1.3)
TARGET_MIX = {
    QueryType.DIRECT: 0.20,
    QueryType.PARAPHRASE: 0.20,
    QueryType.NOTATION: 0.20,
    QueryType.PERSON: 0.15,
    QueryType.TIME_CATEGORY: 0.15,
    QueryType.NO_ANSWER: 0.10,
}
MIX_TOLERANCE = 0.05
DEV_RATIO = 0.4
SECOND_PASS_RATIO = 0.25  # plan requires at least 20%


@dataclass
class GoldSet:
    personas: list[Persona]
    facts: list[GoldFact]
    queries: list[TestQuery]
    labels: list[QueryLabel]


@dataclass
class Issue:
    level: str  # "error" or "warning"
    where: str
    message: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)

    def error(self, where: str, message: str) -> None:
        self.issues.append(Issue("error", where, message))

    def warning(self, where: str, message: str) -> None:
        self.issues.append(Issue("warning", where, message))

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)


def _duplicates(ids: list[str]) -> list[str]:
    return sorted(k for k, n in Counter(ids).items() if n > 1)


def check_gold_set(gs: GoldSet) -> Report:
    r = Report()
    persons = {p.person_id for p in gs.personas}
    facts = {f.fact_id: f for f in gs.facts}
    queries = {q.query_id: q for q in gs.queries}

    for kind, ids in (("persona", [p.person_id for p in gs.personas]), ("fact", [f.fact_id for f in gs.facts]),
                      ("query", [q.query_id for q in gs.queries])):
        for dup in _duplicates(ids):
            r.error(f"{kind} {dup}", "duplicate id")

    for f in gs.facts:
        if f.person_id not in persons:
            r.error(f"fact {f.fact_id}", f"unknown person {f.person_id}")
        if f.supersedes:
            old = facts.get(f.supersedes)
            if old is None:
                r.error(f"fact {f.fact_id}", f"supersedes unknown fact {f.supersedes}")
            elif old.person_id != f.person_id or old.mention_date >= f.mention_date:
                r.error(f"fact {f.fact_id}", f"supersedes {f.supersedes}, which must be an earlier fact of the same person")
    for dup in _duplicates([f"{f.person_id}\n{f.statement}" for f in gs.facts]):
        r.error(f"fact of {dup.splitlines()[0]}", f"duplicate statement: {dup.splitlines()[1]}")

    labels_by_query: dict[str, list[QueryLabel]] = defaultdict(list)
    for dup in _duplicates([f"{l.query_id}/{l.fact_id}/{l.annotator}" for l in gs.labels]):
        r.error(f"label {dup}", "labelled twice by the same annotator")
    for label in gs.labels:
        q, f = queries.get(label.query_id), facts.get(label.fact_id)
        where = f"label {label.query_id}/{label.fact_id}"
        if q is None or f is None:
            r.error(where, "unknown query or fact")
            continue
        if q.person_id != f.person_id:
            r.error(where, "fact belongs to another person; other people's facts are always grade 0")
        if f.mention_date > q.asked_on:
            r.error(where, f"fact is mentioned on {f.mention_date}, after the query is asked ({q.asked_on})")
        flt = q.filters
        if flt.category and f.category != flt.category:
            r.error(where, f"fact category {f.category.value} does not pass the query filter {flt.category.value}")
        if flt.date_from or flt.date_to:
            if f.event_date is None:
                r.error(where, "query has a date filter but the fact has no event_date")
            elif (flt.date_from and f.event_date < flt.date_from) or (flt.date_to and f.event_date > flt.date_to):
                r.error(where, f"fact event_date {f.event_date} is outside the query date filter")
        labels_by_query[label.query_id].append(label)

    for q in gs.queries:
        where = f"query {q.query_id}"
        if q.person_id not in persons:
            r.error(where, f"unknown person {q.person_id}")
        own = labels_by_query.get(q.query_id, [])
        if q.query_type == QueryType.NO_ANSWER:
            if own:
                r.error(where, "no-answer query has relevance labels")
        elif not own:
            r.error(where, "query has no relevant fact")
        elif not any(l.grade == 2 for l in own):
            r.warning(where, "no fact answers it directly (no grade 2)")
        if q.query_type == QueryType.TIME_CATEGORY and q.filters == type(q.filters)():
            r.error(where, "time/category query has no filter")

    mix = Counter(q.query_type for q in gs.queries)
    total = len(gs.queries) or 1
    for qt, target in TARGET_MIX.items():
        share = mix[qt] / total
        if abs(share - target) > MIX_TOLERANCE:
            r.warning("query mix", f"{qt.value} is {share:.0%}, target {target:.0%}")
    return r


def stats(gs: GoldSet) -> dict:
    by_person = Counter(f.person_id for f in gs.facts)
    labelled = Counter(l.query_id for l in gs.labels)
    answerable = [q for q in gs.queries if q.query_type != QueryType.NO_ANSWER]
    return {
        "personas": len(gs.personas),
        "facts": len(gs.facts),
        "facts_per_person": {"min": min(by_person.values(), default=0), "max": max(by_person.values(), default=0)},
        "fact_categories": dict(Counter(f.category.value for f in gs.facts)),
        "facts_superseding": sum(1 for f in gs.facts if f.supersedes),
        "queries": len(gs.queries),
        "query_types": {qt.value: sum(1 for q in gs.queries if q.query_type == qt) for qt in QueryType},
        "splits": dict(Counter(q.split.value for q in gs.queries)),
        "labels": len(gs.labels),
        "labels_per_answerable_query": round(sum(labelled[q.query_id] for q in answerable) / max(len(answerable), 1), 2),
        "grade_counts": dict(Counter(l.grade for l in gs.labels)),
    }


def assign_splits(queries: list[TestQuery], dev_ratio: float = DEV_RATIO) -> dict[str, Split]:
    """Stratified by query type, ordered by a hash of the ID so the result is stable."""
    out: dict[str, Split] = {}
    for qt in QueryType:
        group = sorted((q for q in queries if q.query_type == qt),
                       key=lambda q: hashlib.sha256(q.query_id.encode()).hexdigest())
        n_dev = round(len(group) * dev_ratio)
        for i, q in enumerate(group):
            out[q.query_id] = Split.DEV if i < n_dev else Split.TEST
    return out


def second_pass_sample(queries: list[TestQuery], ratio: float = SECOND_PASS_RATIO) -> list[str]:
    """Query IDs for the independent second annotation, stratified by query type."""
    out: list[str] = []
    for qt in QueryType:
        group = sorted((q.query_id for q in queries if q.query_type == qt),
                       key=lambda qid: hashlib.sha256(f"second:{qid}".encode()).hexdigest())
        out += group[:max(1, round(len(group) * ratio))]
    return sorted(out)


def agreement(gs: GoldSet, first: str, second: str, sample: list[str]) -> dict:
    """Compare two annotators on every (query, fact of the same person) pair in the sample.

    Unlisted pairs count as grade 0, so both annotators are compared on the full grid.
    """
    queries = {q.query_id: q for q in gs.queries}
    facts_by_person: dict[str, list[GoldFact]] = defaultdict(list)
    for f in gs.facts:
        facts_by_person[f.person_id].append(f)
    grade = {(l.annotator, l.query_id, l.fact_id): l.grade for l in gs.labels}

    a, b, disagreements = [], [], []
    for qid in sample:
        q = queries[qid]
        for f in facts_by_person[q.person_id]:
            ga, gb = grade.get((first, qid, f.fact_id), 0), grade.get((second, qid, f.fact_id), 0)
            a.append(ga)
            b.append(gb)
            if ga != gb:
                disagreements.append({"query_id": qid, "fact_id": f.fact_id, first: ga, second: gb})
    either = [(x, y) for x, y in zip(a, b) if x or y]
    return {
        "sample_queries": len(sample),
        "pairs": len(a),
        "exact_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
        "kappa_all_pairs": round(cohen_kappa(a, b), 4),
        "pairs_marked_by_either": len(either),
        "kappa_marked_pairs": round(cohen_kappa([x for x, _ in either], [y for _, y in either]), 4) if either else None,
        "binary_kappa_all_pairs": round(cohen_kappa([x > 0 for x in a], [y > 0 for y in b]), 4),
        "disagreements": disagreements,
    }
