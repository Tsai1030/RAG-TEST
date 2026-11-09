"""Question -> answer from the stored memories (plan 5.1.6).

query embedding → filtered top-k → threshold → context build → LLM answer → citation check.
When no memory passes the threshold the program refuses without calling the LLM; when
some do, the LLM may still refuse (the second layer). A model without a calibrated
threshold skips the first layer and the trace says `uncalibrated`.

Threshold modes: `per_memory` sends only the memories at or above the threshold;
`gate` only decides whether to call the LLM (the top memory passes) and then sends all
top-k, so a lower-scoring memory that updates a fact still reaches the LLM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict, ValidationError

from memrag.application.llm_calls import cache_key
from memrag.domain.errors import ConfigError
from memrag.domain.ports import (
    ChatMessage,
    EmbeddingProvider,
    LLMProvider,
    PromptStore,
    ResponseCache,
    VectorStore,
)
from memrag.domain.schemas.memory import CATEGORY_LABELS_JA, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter


class AnswerOutput(BaseModel):
    """What the LLM must return."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_memory_ids: list[str]


def output_schema() -> dict:
    return AnswerOutput.model_json_schema()


@dataclass(frozen=True)
class AskSettings:
    top_k: int
    threshold: float | None  # None: the model is not calibrated, no first layer
    context_budget_chars: int
    refusal_text: str
    prompt_version: str = "v1"
    threshold_mode: str = "per_memory"  # per_memory | gate


THRESHOLD_MODES = ("per_memory", "gate")


@dataclass
class Context:
    items: list[tuple[MemoryRecord, float]]
    lines: list[str]
    duplicates: list[str] = field(default_factory=list)
    over_budget: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    # answered | llm_refused | below_threshold | no_memory | invalid_output
    reason: str
    citations: list[str]
    invalid_citations: list[str]  # cited but not in the context
    filter: MemoryFilter
    retrieved: list[dict]  # memory_id, score, passed (the threshold let it into the context)
    context: list[str]
    dropped: dict[str, list[str]]
    threshold: float | None
    raw_output: str | None = None
    cached: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    threshold_mode: str = "per_memory"

    @property
    def uncalibrated(self) -> bool:
        return self.threshold is None


def format_memory(m: MemoryRecord) -> str:
    """One context line: `[m_…] 記録日：…／種類：…／日付：…／内容：…` (日付 only when the memory has one)."""
    parts = [f"記録日：{m.conversation_date.isoformat()}", f"種類：{CATEGORY_LABELS_JA[m.category]}"]
    if m.event_date:
        parts.append(f"日付：{m.event_date.isoformat()}")
    return f"[{m.memory_id}] " + "／".join(parts + [f"内容：{m.content}"])


def build_context(scored: list[tuple[MemoryRecord, float]], budget_chars: int) -> Context:
    """Drop memories with the same content, sort by score, then drop from the lowest score
    until the lines fit the budget. Fixed procedure, no LLM (plan 5.1.6)."""
    ctx = Context([], [])
    seen: set[str] = set()
    used = 0
    for m, score in sorted(scored, key=lambda x: -x[1]):
        if m.content in seen:
            ctx.duplicates.append(m.memory_id)
            continue
        seen.add(m.content)
        line = format_memory(m)
        if ctx.over_budget or used + len(line) > budget_chars:
            ctx.over_budget.append(m.memory_id)
            continue
        used += len(line) + 1
        ctx.items.append((m, score))
        ctx.lines.append(line)
    return ctx


def check_citations(output: AnswerOutput, context_ids: set[str], refusal_text: str) -> tuple[list[str], list[str]]:
    """(valid, invalid) cited ids; a refusal must not cite anything."""
    if output.answer.strip() == refusal_text:
        return [], list(output.cited_memory_ids)
    valid = [i for i in dict.fromkeys(output.cited_memory_ids) if i in context_ids]
    return valid, [i for i in output.cited_memory_ids if i not in context_ids]


def ask(question: str, flt: MemoryFilter, today: date, embedder: EmbeddingProvider, store: VectorStore,
        llm: LLMProvider, prompts: PromptStore, settings: AskSettings, cache: ResponseCache | None = None) -> Answer:
    timings: dict[str, float] = {}
    start = clock = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round((now - clock) * 1000, 2)
        clock = now

    vector = embedder.embed_queries([question])[0]
    lap("embed_query")
    hits = store.search(vector, settings.top_k, flt)
    records = {r.memory_id: r for r in store.get([h.memory_id for h in hits])}
    lap("retrieve")
    t = settings.threshold
    if settings.threshold_mode not in THRESHOLD_MODES:
        raise ConfigError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {settings.threshold_mode!r}")
    if t is None:
        passed = [True] * len(hits)
    elif settings.threshold_mode == "gate":
        passed = [bool(hits) and hits[0].score >= t] * len(hits)
    else:
        passed = [h.score >= t for h in hits]
    retrieved = [{"memory_id": h.memory_id, "score": round(h.score, 4), "passed": p} for h, p in zip(hits, passed)]
    ctx = build_context([(records[h.memory_id], h.score) for h, p in zip(hits, passed) if p],
                        settings.context_budget_chars)
    lap("context")
    answer = Answer(question, settings.refusal_text, True, "no_memory" if not hits else "below_threshold", [], [],
                    flt, retrieved, ctx.lines, {"duplicate": ctx.duplicates, "over_budget": ctx.over_budget}, t,
                    threshold_mode=settings.threshold_mode)
    if ctx.items:
        schema = output_schema()
        system, user = prompts.get("answer", settings.prompt_version).render(
            today=today.isoformat(), question=question, memories="\n".join(ctx.lines),
            refusal_text=settings.refusal_text, output_schema=json.dumps(schema, ensure_ascii=False))
        messages: list[ChatMessage] = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        key = cache_key(messages, llm, schema)
        cached = cache.get(key) if cache else None
        raw = cached if cached is not None else llm.chat(messages, schema)
        lap("llm")
        answer.raw_output, answer.cached = raw, cached is not None
        try:
            output = AnswerOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            answer.reason = "invalid_output"
        else:
            if cache and cached is None:
                cache.put(key, raw)
            valid, invalid = check_citations(output, {m.memory_id for m, _ in ctx.items}, settings.refusal_text)
            answer.refused = output.answer.strip() == settings.refusal_text
            answer.text = settings.refusal_text if answer.refused else output.answer.strip()
            answer.reason = "llm_refused" if answer.refused else "answered"
            answer.citations, answer.invalid_citations = valid, invalid
    timings["total"] = round((time.perf_counter() - start) * 1000, 2)
    answer.timings_ms = timings
    return answer
