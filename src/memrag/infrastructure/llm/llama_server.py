"""LLMProvider backed by a llama.cpp `llama-server` (OpenAI-compatible API), standard library only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Sequence

from memrag.domain.errors import LLMUnavailableError
from memrag.domain.ports import ChatMessage

NETWORK_RETRIES = 4  # exponential backoff: 1, 2, 4, 8 s


class LlamaServerProvider:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0, seed: int = 42,
                 max_tokens: int = 2048, timeout_s: float = 600):
        self._url = base_url.rstrip("/")
        self._params = {"model": model, "temperature": temperature, "seed": seed, "max_tokens": max_tokens}
        self._timeout = timeout_s

    def info(self) -> dict:
        return dict(self._params)

    def served_model(self) -> str:
        """File name of the model the server actually loaded, to check it matches the config."""
        with urllib.request.urlopen(f"{self._url}/v1/models", timeout=30) as resp:
            return json.load(resp)["data"][0]["id"]

    def chat(self, messages: Sequence[ChatMessage], json_schema: dict) -> str:
        body = {
            "messages": list(messages),
            "temperature": self._params["temperature"],
            "seed": self._params["seed"],
            "max_tokens": self._params["max_tokens"],
            "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema}},
            # Qwen3 hybrid models think by default; summaries must not
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(f"{self._url}/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
        for attempt in range(NETWORK_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == NETWORK_RETRIES:
                    raise LLMUnavailableError(f"llama-server at {self._url} failed: {e}") from e
                time.sleep(2**attempt)
        raise AssertionError("unreachable")
