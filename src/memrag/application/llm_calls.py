"""Shared by every LLM call: the response-cache key."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from memrag.domain.ports import ChatMessage, LLMProvider


def cache_key(messages: Sequence[ChatMessage], llm: LLMProvider, schema: dict) -> str:
    """Hash of everything that affects the output: messages, model and sampling, output schema."""
    payload = json.dumps({"messages": list(messages), "llm": llm.info(), "schema": schema},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
