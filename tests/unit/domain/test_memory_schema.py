from datetime import date

import pytest
from pydantic import ValidationError

from memrag.domain.schemas.memory import Category, MemoryDraft, MemoryRecord


def draft(**overrides) -> MemoryDraft:
    values = dict(category="plan", content="佐藤さんは11月15日に孫の結衣ちゃんが泊まりに来る予定だ。",
                  event_date="2026-11-15", people=["孫の結衣"], source_turn_ids=[3, 4])
    return MemoryDraft(**(values | overrides))


def test_valid_draft_parses_enum_and_date():
    d = draft()
    assert d.category is Category.PLAN
    assert d.event_date == date(2026, 11, 15)


@pytest.mark.parametrize("content", ["", "   ", "這是中文沒有假名。", "あ" * 201])
def test_content_rules(content):
    with pytest.raises(ValidationError):
        draft(content=content)


def test_unknown_field_and_category_are_rejected():
    with pytest.raises(ValidationError):
        draft(person_id="p01")  # filled by the program, not the LLM
    with pytest.raises(ValidationError):
        draft(category="fact")


def test_source_turns_required():
    with pytest.raises(ValidationError):
        draft(source_turn_ids=[])


def test_record_adds_program_fields():
    r = MemoryRecord.from_draft(draft(), person_id="p01", conversation_date=date(2026, 10, 14), source_log_id="log_p01_1014")
    assert r.event_date_ord == 20261115
    assert r.memory_id.startswith("m_") and len(r.memory_id) == 14
    assert r.schema_version == "1"
    assert r.memory_id == MemoryRecord.from_draft(draft(), person_id="p01", conversation_date=date(2026, 10, 14),
                                                  source_log_id="log_p01_1014").memory_id


def test_record_without_event_date():
    r = MemoryRecord.from_draft(draft(category="preference", event_date=None), person_id="p01",
                                conversation_date=date(2026, 10, 2), source_log_id="log")
    assert r.event_date_ord is None
