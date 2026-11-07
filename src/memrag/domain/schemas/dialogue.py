"""One day's conversation between the companion robot and one user."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Speaker(str, Enum):
    USER = "user"
    ROBOT = "robot"


SPEAKER_LABELS_JA = {Speaker.USER: "ユーザー", Speaker.ROBOT: "ロボット"}


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: int = Field(ge=1)
    speaker: Speaker
    text: str

    @field_validator("text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("turn text is empty")
        return value.strip()


class DialogueLog(BaseModel):
    """`fact_turns` maps each gold fact told in this log to its turns; evaluation only, never shown to the LLM."""

    model_config = ConfigDict(extra="forbid")

    log_id: str
    person_id: str
    date: date
    turns: list[Turn] = Field(min_length=1)
    fact_turns: dict[str, list[int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _turn_ids(self) -> DialogueLog:
        ids = [t.turn_id for t in self.turns]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"{self.log_id}: turn ids must be 1..{len(ids)} in order")
        for fact_id, turns in self.fact_turns.items():
            if not turns or any(t not in ids for t in turns):
                raise ValueError(f"{self.log_id}: fact {fact_id} points to missing turns {turns}")
        return self

    def render(self) -> str:
        """The log as the LLM sees it: one numbered line per turn."""
        return "\n".join(f"[{t.turn_id}] {SPEAKER_LABELS_JA[t.speaker]}：{t.text}" for t in self.turns)
