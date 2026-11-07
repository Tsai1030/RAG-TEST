"""Checks on the dialogue logs, and the 5.1.4 acceptance numbers for a summarized memory corpus."""

from __future__ import annotations

from collections import Counter

from memrag.application.evaluation.dataset_check import GoldSet, Report
from memrag.application.summarize import SummaryResult
from memrag.domain.schemas.dialogue import DialogueLog, Speaker


def check_dialogues(gs: GoldSet, logs: list[DialogueLog]) -> Report:
    """Every gold fact is told exactly once, by the right person, on its mention date."""
    report = Report()
    facts = {f.fact_id: f for f in gs.facts}
    persons = {p.person_id for p in gs.personas}
    for log_id, n in Counter(lg.log_id for lg in logs).items():
        if n > 1:
            report.error(log_id, f"log_id 重複 {n} 次")
    told: Counter[str] = Counter()
    for lg in logs:
        if lg.person_id not in persons:
            report.error(lg.log_id, f"未知的使用者 {lg.person_id}")
        if not any(t.speaker == Speaker.USER for t in lg.turns):
            report.error(lg.log_id, "沒有使用者的發言")
        for fact_id in lg.fact_turns:
            told[fact_id] += 1
            fact = facts.get(fact_id)
            if fact is None:
                report.error(lg.log_id, f"未知的事實 {fact_id}")
            elif fact.person_id != lg.person_id:
                report.error(lg.log_id, f"{fact_id} 屬於 {fact.person_id}")
            elif fact.mention_date != lg.date:
                report.error(lg.log_id, f"{fact_id} 應在 {fact.mention_date} 提到，對話日期是 {lg.date}")
    for fact_id in facts:
        if told[fact_id] == 0:
            report.error(fact_id, "沒有出現在任何對話")
        elif told[fact_id] > 1:
            report.error(fact_id, f"出現在 {told[fact_id]} 場對話")
    return report


def corpus_stats(logs: list[DialogueLog], results: list[SummaryResult]) -> dict:
    """Success rates, gold fact coverage (uncovered facts listed) and memory counts.

    Coverage counts the facts told in the summarized logs; facts in failed logs count as uncovered.
    """
    n = len(results)
    records = [r for res in results for r in res.records]
    covered = {f for r in records for f in r.gold_fact_ids}
    told = sorted({f for lg in logs for f in lg.fact_turns})
    uncovered = [f for f in told if f not in covered]
    retries = Counter(len(res.attempts) - 1 for res in results)
    return {
        "logs": n,
        "success_rate": round(sum(res.ok for res in results) / n, 4) if n else None,
        "first_pass_rate": round(sum(res.first_pass_ok for res in results) / n, 4) if n else None,
        "retries": {str(k): v for k, v in sorted(retries.items())},
        "failed_logs": [res.log_id for res in results if not res.ok],
        "memories": len(records),
        "memories_without_gold_fact": sum(not r.gold_fact_ids for r in records),
        "gold_fact_coverage": round(1 - len(uncovered) / len(told), 4) if told else None,
        "uncovered_facts": uncovered,
    }
