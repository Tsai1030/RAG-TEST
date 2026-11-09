"""Retrieval filters and hits.

`MemoryFilter.matches` is the reference meaning of a filter: every VectorStore must
return exactly the memories it accepts. A date bound excludes memories without an
event date, unless `date_fallback` is set: then such a memory is matched by the day
it was told (conversation_date). The pipeline uses the fallback (decided in 5.1.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, model_validator

from memrag.domain.schemas.memory import Category, MemoryRecord


class MemoryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    person_id: str | None = None
    category: Category | None = None
    date_from: date | None = None
    date_to: date | None = None
    date_fallback: bool = False

    @model_validator(mode="after")
    def _range(self) -> MemoryFilter:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        return self

    @property
    def has_date(self) -> bool:
        return self.date_from is not None or self.date_to is not None

    def matches(self, m: MemoryRecord) -> bool:
        if self.person_id is not None and m.person_id != self.person_id:
            return False
        if self.category is not None and m.category != self.category:
            return False
        if self.has_date:
            day = m.event_date or (m.conversation_date if self.date_fallback else None)
            if day is None:
                return False
            if self.date_from and day < self.date_from:
                return False
            if self.date_to and day > self.date_to:
                return False
        return True


@dataclass(frozen=True)
class SearchHit:
    memory_id: str
    score: float  # cosine similarity
