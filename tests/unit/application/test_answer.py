import json
from datetime import date

import numpy as np
import pytest

from memrag.application.answer import (
    AnswerOutput,
    AskSettings,
    ask,
    build_context,
    check_citations,
    format_memory,
)
from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord
from memrag.domain.schemas.prompt import PromptTemplate
from memrag.domain.schemas.retrieval import MemoryFilter
from memrag.infrastructure.llm.fake import FakeLLMProvider
from memrag.infrastructure.vectorstore.exact_store import ExactStore

REFUSAL = "関連する情報はありません"


def _rec(person, content, told, category="preference", event_date=None, log="d"):
    d = MemoryDraft(category=category, content=content, event_date=event_date, source_turn_ids=[1])
    return MemoryRecord.from_draft(d, person_id=person, conversation_date=told, source_log_id=log)


OLD = _rec("p06", "渡辺さんはとんこつラーメンが大好きで週に三回食べる。", date(2026, 10, 2))
NEW = _rec("p06", "渡辺さんは健康診断のあと、ラーメンを月に二回までにした。", date(2026, 10, 27), "health")
TRIP = _rec("p06", "渡辺さんは11月29日に横浜のハーフマラソンに出る。", date(2026, 10, 28), "plan", date(2026, 11, 29))
OTHER = _rec("p01", "佐藤さんはラーメンが苦手。", date(2026, 10, 3))


class Prompts:
    def get(self, name, version):
        return PromptTemplate(name, version, "draft", ("today", "question", "memories", "refusal_text", "output_schema"),
                              "sys {{refusal_text}}", "{{today}}\n{{question}}\n{{memories}}")


class Embedder:
    """Queries embed to the first axis; the store's vectors decide the scores."""

    def embed_queries(self, texts):
        return [np.array([1.0, 0.0, 0.0])]


def _store():
    store = ExactStore(3)
    store.upsert([OLD, NEW, TRIP, OTHER], np.array([[0.95, 0.312, 0], [0.9, 0, 0.436], [0.3, 0.954, 0], [0.99, 0.141, 0]]))
    return store


P06 = MemoryFilter(person_id="p06")


def _ask(llm, threshold=0.85, flt=P06, mode="per_memory"):
    settings = AskSettings(top_k=5, threshold=threshold, context_budget_chars=1200, refusal_text=REFUSAL,
                           threshold_mode=mode)
    return ask("ラーメン、今も食べてる？", flt, date(2026, 11, 1), Embedder(), _store(), llm, Prompts(), settings)


def _out(answer, cited):
    return json.dumps({"answer": answer, "cited_memory_ids": cited}, ensure_ascii=False)


def test_format_memory_shows_told_date_and_event_date():
    assert format_memory(NEW) == f"[{NEW.memory_id}] 記録日：2026-10-27／種類：健康・体調／内容：{NEW.content}"
    assert "／日付：2026-11-29／" in format_memory(TRIP)


def test_build_context_dedups_sorts_and_drops_lowest_over_budget():
    dup = _rec("p06", OLD.content, date(2026, 10, 9), log="d2")
    ctx = build_context([(NEW, 0.9), (dup, 0.8), (OLD, 0.95), (TRIP, 0.3)], budget_chars=10_000)
    assert [m.memory_id for m, _ in ctx.items] == [OLD.memory_id, NEW.memory_id, TRIP.memory_id]
    assert ctx.duplicates == [dup.memory_id]
    budget = len(format_memory(OLD)) + 1 + len(format_memory(NEW))
    ctx = build_context([(NEW, 0.9), (OLD, 0.95), (TRIP, 0.3)], budget_chars=budget)
    assert [m.memory_id for m, _ in ctx.items] == [OLD.memory_id, NEW.memory_id]
    assert ctx.over_budget == [TRIP.memory_id]


def test_check_citations():
    ids = {OLD.memory_id}
    assert check_citations(AnswerOutput(answer="はい", cited_memory_ids=[OLD.memory_id, "m_x", OLD.memory_id]), ids,
                           REFUSAL) == ([OLD.memory_id], ["m_x"])
    # A refusal must not cite anything
    assert check_citations(AnswerOutput(answer=REFUSAL, cited_memory_ids=[OLD.memory_id]), ids, REFUSAL) == (
        [], [OLD.memory_id])


def test_answer_with_citations_inside_the_filter():
    llm = FakeLLMProvider([_out("月に二回までにしたそうですね。", [NEW.memory_id])])
    a = _ask(llm)
    assert (a.refused, a.reason, a.citations, a.invalid_citations) == (False, "answered", [NEW.memory_id], [])
    assert [h["memory_id"] for h in a.retrieved] == [OLD.memory_id, NEW.memory_id, TRIP.memory_id]  # never p01
    assert [h["passed"] for h in a.retrieved] == [True, True, False]
    prompt = llm.requests[0][1]["content"]
    assert prompt.startswith("2026-11-01") and TRIP.memory_id not in prompt
    assert set(a.timings_ms) == {"embed_query", "retrieve", "context", "prompt", "llm", "parse", "total"}


def test_no_memory_above_threshold_refuses_without_the_llm():
    llm = FakeLLMProvider([])
    a = _ask(llm, threshold=0.99)
    assert (a.refused, a.reason, a.text) == (True, "below_threshold", REFUSAL)
    assert llm.requests == [] and "llm" not in a.timings_ms
    a = _ask(llm, flt=MemoryFilter(person_id="p99"))
    assert a.reason == "no_memory" and llm.requests == []


def test_llm_refusal_and_invalid_output():
    a = _ask(FakeLLMProvider([_out(REFUSAL, [])]))
    assert (a.refused, a.reason, a.citations) == (True, "llm_refused", [])
    a = _ask(FakeLLMProvider(["not json"]))
    assert (a.refused, a.reason, a.raw_output) == (True, "invalid_output", "not json")


def test_gate_mode_sends_all_top_k_when_the_top_memory_passes():
    # Scores: OLD 0.95, NEW 0.90, TRIP 0.30. Per memory at 0.92 the newer fact (NEW) is cut off.
    llm = FakeLLMProvider([_out("はい", [OLD.memory_id])])
    a = _ask(llm, threshold=0.92)
    assert [h["passed"] for h in a.retrieved] == [True, False, False] and NEW.memory_id not in llm.requests[0][1]["content"]
    llm = FakeLLMProvider([_out("月に二回までにしたそうですね。", [NEW.memory_id])])
    a = _ask(llm, threshold=0.92, mode="gate")
    assert all(h["passed"] for h in a.retrieved) and a.threshold_mode == "gate"
    assert NEW.memory_id in llm.requests[0][1]["content"] and a.citations == [NEW.memory_id]
    # The gate still refuses without the LLM when even the top memory is below the threshold
    llm = FakeLLMProvider([])
    a = _ask(llm, threshold=0.99, mode="gate")
    assert (a.reason, llm.requests) == ("below_threshold", [])
    assert not any(h["passed"] for h in a.retrieved)


def test_unknown_threshold_mode_is_a_config_error():
    from memrag.domain.errors import ConfigError

    with pytest.raises(ConfigError):
        _ask(FakeLLMProvider([]), mode="nope")


def test_uncalibrated_model_skips_the_threshold():
    llm = FakeLLMProvider([_out("はい", ["m_000000000000"])])
    a = _ask(llm, threshold=None)
    assert a.uncalibrated and all(h["passed"] for h in a.retrieved)
    assert a.citations == [] and a.invalid_citations == ["m_000000000000"]


@pytest.mark.parametrize("flt, expected", [
    (MemoryFilter(person_id="p06", date_from=date(2026, 10, 20), date_fallback=True), {NEW.memory_id, TRIP.memory_id}),
    (MemoryFilter(person_id="p06", date_from=date(2026, 10, 20)), {TRIP.memory_id}),
])
def test_date_filter_with_fallback(flt, expected):
    a = _ask(FakeLLMProvider([_out(REFUSAL, [])]), threshold=None, flt=flt)
    assert {h["memory_id"] for h in a.retrieved} == expected
