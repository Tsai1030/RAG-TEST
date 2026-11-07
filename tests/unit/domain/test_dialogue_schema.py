from datetime import date

import pytest
from pydantic import ValidationError

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.dialogue import DialogueLog
from memrag.domain.schemas.prompt import PromptTemplate


def _log(**overrides):
    data = {
        "log_id": "d_p01_1001", "person_id": "p01", "date": date(2026, 10, 1),
        "turns": [{"turn_id": 1, "speaker": "robot", "text": "おはようございます。"},
                  {"turn_id": 2, "speaker": "user", "text": "おはよう。今日は寒いわね。"}],
        "fact_turns": {"f_p01_01": [2]},
    }
    return DialogueLog.model_validate({**data, **overrides})


def test_render_numbers_turns_and_hides_fact_tags():
    text = _log().render()
    assert text == "[1] ロボット：おはようございます。\n[2] ユーザー：おはよう。今日は寒いわね。"
    assert "f_p01_01" not in text


def test_turn_ids_must_be_consecutive():
    with pytest.raises(ValidationError):
        _log(turns=[{"turn_id": 1, "speaker": "user", "text": "あ"}, {"turn_id": 3, "speaker": "user", "text": "い"}])


def test_fact_turns_must_exist():
    with pytest.raises(ValidationError):
        _log(fact_turns={"f_p01_01": [5]})
    with pytest.raises(ValidationError):
        _log(fact_turns={"f_p01_01": []})


def test_prompt_render_checks_variables():
    p = PromptTemplate("x", "v1", "draft", ("a", "b"), "sys {{a}}", "user {{b}}")
    assert p.render(a="1", b="2") == ("sys 1", "user 2")
    with pytest.raises(ConfigError):
        p.render(a="1")
    with pytest.raises(ConfigError):
        p.render(a="1", b="2", c="3")
    assert PromptTemplate("x", "v1", "draft", (), None, "u").render() == (None, "u")
    assert p.sha256 == PromptTemplate("x", "v2", "frozen", ("a", "b"), "sys {{a}}", "user {{b}}").sha256
