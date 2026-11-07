"""Memory schema: what the LLM extracts from one day's conversation (D12).

The LLM fills MemoryDraft only. The program adds who the user is, which day
the conversation was, and IDs; evaluation adds gold_fact_ids, which the LLM
never sees.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1"
MAX_CONTENT_CHARS = 200
_KANA = re.compile(r"[ぁ-ゟ゠-ヿ]")


class Category(str, Enum):
    EVENT = "event"
    PLAN = "plan"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    HEALTH = "health"
    EMOTION = "emotion"
    ROUTINE = "routine"
    PROFILE = "profile"


CATEGORY_LABELS_JA = {
    Category.EVENT: "出来事",
    Category.PLAN: "予定",
    Category.PREFERENCE: "好み",
    Category.RELATIONSHIP: "人間関係",
    Category.HEALTH: "健康・体調",
    Category.EMOTION: "気持ち",
    Category.ROUTINE: "習慣",
    Category.PROFILE: "プロフィール",
}


def check_content(text: str) -> str:
    """Shared rules for memory text: Japanese (has kana), non-empty, at most 200 characters."""
    text = text.strip()
    if not text:
        raise ValueError("content is empty")
    if len(text) > MAX_CONTENT_CHARS:
        raise ValueError(f"content has {len(text)} characters, limit is {MAX_CONTENT_CHARS}")
    if not _KANA.search(text):
        raise ValueError("content has no kana; it must be Japanese")
    return text


class MemoryDraft(BaseModel):
    """LLM output for one memory."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    content: str
    event_date: date | None = None
    people: list[str] = Field(default_factory=list)
    source_turn_ids: list[int] = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _content_rules(cls, value: str) -> str:
        return check_content(value)


class MemoryRecord(MemoryDraft):
    """A stored memory: the draft plus fields the program fills in."""

    person_id: str
    conversation_date: date
    source_log_id: str
    memory_id: str
    event_date_ord: int | None = None
    schema_version: str = SCHEMA_VERSION
    gold_fact_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_draft(cls, draft: MemoryDraft, *, person_id: str, conversation_date: date, source_log_id: str) -> MemoryRecord:
        digest = hashlib.sha256(f"{person_id}\n{source_log_id}\n{draft.content}".encode("utf-8")).hexdigest()
        return cls(
            **draft.model_dump(),
            person_id=person_id,
            conversation_date=conversation_date,
            source_log_id=source_log_id,
            memory_id=f"m_{digest[:12]}",
            event_date_ord=int(draft.event_date.strftime("%Y%m%d")) if draft.event_date else None,
        )
