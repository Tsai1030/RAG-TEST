from datetime import date

from memrag.application.evaluation.corpus_check import check_dialogues, corpus_stats
from memrag.application.evaluation.dataset_check import GoldSet
from memrag.application.summarize import Attempt, SummaryResult
from memrag.domain.schemas.dialogue import DialogueLog
from memrag.domain.schemas.evaluation import GoldFact, Persona
from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord

FACTS = [
    GoldFact(fact_id="f_p01_01", person_id="p01", category="profile", statement="佐藤さんは川越に住む。",
             mention_date=date(2026, 10, 1)),
    GoldFact(fact_id="f_p01_02", person_id="p01", category="routine", statement="佐藤さんは毎朝散歩する。",
             mention_date=date(2026, 10, 2)),
]
GS = GoldSet(personas=[Persona(person_id="p01", name="佐藤 和子", age=78, household="-", speech_style="-", profile="-")],
             facts=FACTS, queries=[], labels=[])


def _log(log_id, day, facts):
    return DialogueLog(log_id=log_id, person_id="p01", date=date(2026, 10, day),
                       turns=[{"turn_id": 1, "speaker": "user", "text": "そうなのよ。"}],
                       fact_turns={f: [1] for f in facts})


def test_check_dialogues_ok_and_errors():
    assert check_dialogues(GS, [_log("a", 1, ["f_p01_01"]), _log("b", 2, ["f_p01_02"])]).ok
    report = check_dialogues(GS, [_log("a", 1, ["f_p01_01", "f_p01_02"]), _log("a", 3, ["f_p01_01"])])
    messages = " ".join(f"{i.where} {i.message}" for i in report.issues)
    assert "log_id 重複" in messages
    assert "應在 2026-10-02" in messages
    assert "f_p01_01 出現在 2 場對話" in messages


def _record(content, facts):
    draft = MemoryDraft(category="event", content=content, source_turn_ids=[1])
    rec = MemoryRecord.from_draft(draft, person_id="p01", conversation_date=date(2026, 10, 1), source_log_id="a")
    return rec.model_copy(update={"gold_fact_ids": facts})


def test_corpus_stats_counts_coverage_and_failures():
    ok = SummaryResult("a", [_record("佐藤さんは川越に住む。", ["f_p01_01"]), _record("天気がいい。", [])],
                       [Attempt("x", ["bad"]), Attempt("y", [])])
    failed = SummaryResult("b", [], [Attempt("z", ["bad"])] * 4)
    stats = corpus_stats([_log("a", 1, ["f_p01_01"]), _log("b", 2, ["f_p01_02"])], [ok, failed])
    assert stats["success_rate"] == 0.5 and stats["first_pass_rate"] == 0.0
    assert stats["failed_logs"] == ["b"]
    assert stats["memories"] == 2 and stats["memories_without_gold_fact"] == 1
    assert stats["gold_fact_coverage"] == 0.5 and stats["uncovered_facts"] == ["f_p01_02"]
    assert stats["retries"] == {"1": 1, "3": 1}
