"""The real prompt files load, declare exactly the variables they use, and render."""

import re
from pathlib import Path

import pytest

from memrag.domain.errors import ConfigError
from memrag.infrastructure.prompts import YamlPromptStore

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"


@pytest.mark.parametrize("name", ["summarize", "summarize_retry", "dialogue_generate", "answer"])
def test_prompt_variables_match_placeholders(name):
    p = YamlPromptStore(PROMPTS).get(name, "v1")
    used = set(re.findall(r"\{\{(\w+)\}\}", (p.system or "") + p.user))
    assert used == set(p.variables)
    system, user = p.render(**{v: f"<{v}>" for v in p.variables})
    assert "{{" not in (system or "") + user


def test_missing_or_mismatched_prompt(tmp_path):
    with pytest.raises(ConfigError):
        YamlPromptStore(tmp_path).get("summarize", "v1")
    (tmp_path / "x.v1.yaml").write_text("name: y\nversion: v1\nuser: hi\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        YamlPromptStore(tmp_path).get("x", "v1")
