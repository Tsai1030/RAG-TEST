"""Evaluation data: simulated users, gold facts, test queries and relevance labels.

Facts come first: dialogues are generated from them (5.1.4), and queries are
labelled against them, so the ground truth never depends on the summarizer.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memrag.domain.schemas.memory import Category, check_content


class QueryType(str, Enum):
    DIRECT = "direct"            # 直接事實
    PARAPHRASE = "paraphrase"    # 改寫／同義
    NOTATION = "notation"        # 日文表記與語體
    PERSON = "person"            # 人物限定：相似內容屬於不同人
    TIME_CATEGORY = "time_category"  # 時間與類型限定
    NO_ANSWER = "no_answer"      # 無答案


class Split(str, Enum):
    DEV = "dev"
    TEST = "test"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Persona(_Strict):
    """A simulated companion-robot user. Everything here is fictional."""

    person_id: str
    name: str
    age: int
    household: str
    speech_style: str
    profile: str


class GoldFact(_Strict):
    """One fact the user tells the robot on a given day; written like a memory."""

    fact_id: str
    person_id: str
    category: Category
    statement: str
    event_date: date | None = None
    people: list[str] = Field(default_factory=list)
    mention_date: date
    supersedes: str | None = None

    @field_validator("statement")
    @classmethod
    def _statement_rules(cls, value: str) -> str:
        return check_content(value)


class QueryFilter(_Strict):
    date_from: date | None = None
    date_to: date | None = None
    category: Category | None = None

    @model_validator(mode="after")
    def _range(self) -> QueryFilter:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        return self


class TestQuery(_Strict):
    """A user utterance the robot must find memories for."""

    __test__ = False  # not a pytest test class

    query_id: str
    person_id: str
    text: str
    query_type: QueryType
    filters: QueryFilter = Field(default_factory=QueryFilter)
    asked_on: date
    split: Split
    note: str = ""

    @field_validator("text")
    @classmethod
    def _text_rules(cls, value: str) -> str:
        return check_content(value)


class QueryLabel(_Strict):
    """Graded relevance of one gold fact to one query; unlisted pairs are grade 0."""

    query_id: str
    fact_id: str
    grade: int = Field(ge=1, le=2)
    annotator: str
