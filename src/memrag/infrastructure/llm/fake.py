"""LLMProvider for tests and --fake runs, no model needed.

With scripted responses it returns them in order. Without them it echoes:
every user turn in the dialogue log becomes one `event` memory citing that turn.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from memrag.domain.ports import ChatMessage

_USER_TURN = re.compile(r"^\[(\d+)\] ユーザー：(.+)$", re.MULTILINE)


class FakeLLMProvider:
    def __init__(self, responses: Sequence[str] | None = None):
        self._responses = list(responses) if responses is not None else None
        self.requests: list[list[ChatMessage]] = []

    def info(self) -> dict:
        return {"model": "fake", "temperature": 0.0, "seed": 0}

    def chat(self, messages: Sequence[ChatMessage], json_schema: dict) -> str:
        self.requests.append(list(messages))
        if self._responses is not None:
            return self._responses.pop(0) if self._responses else '{"memories": []}'
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        memories = [{"category": "event", "content": text[:200], "event_date": None, "people": [],
                     "source_turn_ids": [int(n)]} for n, text in _USER_TURN.findall(prompt)]
        return json.dumps({"memories": memories}, ensure_ascii=False)
