"""Dialogue log -> memory records: structured LLM output, validation, retries (plan 5.1.4)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, ValidationError

from memrag.domain.ports import ChatMessage, LLMProvider, PromptStore, ResponseCache
from memrag.domain.schemas.dialogue import DialogueLog
from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord

MAX_RETRIES = 3
EVENT_DATE_WINDOW = timedelta(days=400)  # an event this far from the conversation is almost surely a date error
_HANDLE = re.compile(r"^(.+?(?:さん|ちゃん|くん))(?:は|の|が)")


def user_handle(statements: list[str], fallback: str) -> str:
    """How memories name the user (佐藤さん, さくらちゃん), taken from the gold facts' wording."""
    found = Counter(m.group(1) for s in statements if (m := _HANDLE.match(s)))
    return found.most_common(1)[0][0] if found else fallback


class SummaryOutput(BaseModel):
    """What the LLM must return: a list of memories, possibly empty."""

    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryDraft]


def output_schema() -> dict:
    return SummaryOutput.model_json_schema()


@dataclass
class Attempt:
    raw: str
    errors: list[str]
    cached: bool = False


@dataclass
class SummaryResult:
    log_id: str
    records: list[MemoryRecord] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.attempts) and not self.attempts[-1].errors

    @property
    def first_pass_ok(self) -> bool:
        return bool(self.attempts) and not self.attempts[0].errors


def validate_output(raw: str, log: DialogueLog) -> tuple[list[MemoryDraft], list[str]]:
    """Schema and semantic checks. Returns the drafts (deduplicated) and the errors, written in Japanese for the LLM."""
    try:
        parsed = SummaryOutput.model_validate(json.loads(raw))
    except json.JSONDecodeError as e:
        return [], [f"JSON として読めません：{e}"]
    except ValidationError as e:
        return [], [f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors()]
    errors = []
    turn_ids = {t.turn_id for t in log.turns}
    drafts: list[MemoryDraft] = []
    seen: set[str] = set()
    for i, d in enumerate(parsed.memories):
        bad = sorted(set(d.source_turn_ids) - turn_ids)
        if bad:
            errors.append(f"memories.{i}.source_turn_ids：存在しないターン番号 {bad}")
        if d.event_date and abs(d.event_date - log.date) > EVENT_DATE_WINDOW:
            errors.append(f"memories.{i}.event_date：{d.event_date} は会話の日付 {log.date} から離れすぎています")
        if d.content not in seen:
            seen.add(d.content)
            drafts.append(d)
    return drafts, errors


def gold_facts_for(draft: MemoryDraft, log: DialogueLog) -> list[str]:
    """Gold facts whose turns the memory cites. Evaluation only."""
    cited = set(draft.source_turn_ids)
    return sorted(f for f, turns in log.fact_turns.items() if cited.intersection(turns))


def _cache_key(messages: list[ChatMessage], llm: LLMProvider, schema: dict) -> str:
    payload = json.dumps({"messages": messages, "llm": llm.info(), "schema": schema}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_log(log: DialogueLog, user_name: str, llm: LLMProvider, prompts: PromptStore,
                  cache: ResponseCache | None = None, version: str = "v1") -> SummaryResult:
    schema = output_schema()
    system, user = prompts.get("summarize", version).render(
        log_date=log.date.isoformat(), user_name=user_name, dialogue_log=log.render(),
        output_schema=json.dumps(schema, ensure_ascii=False))
    messages: list[ChatMessage] = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}]
    retry = prompts.get("summarize_retry", version)
    result = SummaryResult(log.log_id)
    for _ in range(1 + MAX_RETRIES):
        key = _cache_key(messages, llm, schema)
        cached = cache.get(key) if cache else None
        raw = cached if cached is not None else llm.chat(messages, schema)
        drafts, errors = validate_output(raw, log)
        result.attempts.append(Attempt(raw, errors, cached is not None))
        if not errors:
            if cache and cached is None:
                cache.put(key, raw)
            result.records = [
                MemoryRecord.from_draft(d, person_id=log.person_id, conversation_date=log.date,
                                        source_log_id=log.log_id).model_copy(
                    update={"gold_fact_ids": gold_facts_for(d, log)})
                for d in drafts
            ]
            return result
        _, retry_user = retry.render(previous_output=raw, validation_errors="\n".join(errors))
        messages = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry_user}]
    return result
