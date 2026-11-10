"""LLMProvider backed by a llama.cpp `llama-server` (OpenAI-compatible API), standard library only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Sequence

from memrag.domain.errors import LLMUnavailableError
from memrag.domain.ports import ChatMessage

NETWORK_RETRIES = 4  # exponential backoff: 1, 2, 4, 8 s


def call_stats(response: dict, wall_ms: float, retries: int = 0) -> dict:
    """One completion's token counts and llama-server's own timings (prefill = prompt processing)."""
    t = response.get("timings") or {}
    u = response.get("usage") or {}
    return {"wall_ms": round(wall_ms, 2), "retries": retries, "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "cached_tokens": t.get("cache_n"),
            "prefill_tokens": t.get("prompt_n"), "prefill_ms": t.get("prompt_ms"),
            "decode_tokens": t.get("predicted_n"), "decode_ms": t.get("predicted_ms")}


class LlamaServerProvider:
    """Set `call_log` to a list to collect `call_stats` of every completion (profiling)."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0, seed: int = 42,
                 max_tokens: int = 2048, timeout_s: float = 600):
        self._url = base_url.rstrip("/")
        self._params = {"model": model, "temperature": temperature, "seed": seed, "max_tokens": max_tokens}
        self._timeout = timeout_s
        self.call_log: list[dict] | None = None

    def info(self) -> dict:
        return dict(self._params)

    def served_model(self) -> str:
        """File name of the model the server actually loaded, to check it matches the config."""
        with urllib.request.urlopen(f"{self._url}/v1/models", timeout=30) as resp:
            return json.load(resp)["data"][0]["id"]

    def server_rss(self) -> int | None:
        """Resident memory in bytes of the local llama-server on this port; None when not found.
        On the Jetson the GPU-offloaded weights are not in RSS (see MemAvailable instead)."""
        import psutil

        port = str(urllib.parse.urlparse(self._url).port)
        for p in psutil.process_iter(["name", "cmdline"]):
            cmd = p.info["cmdline"] or []
            if "llama-server" in (p.info["name"] or "") and "--port" in cmd[:-1] and cmd[cmd.index("--port") + 1] == port:
                try:
                    return p.memory_info().rss
                except psutil.Error:
                    return None
        return None

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
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                    data = json.load(resp)
                if self.call_log is not None:
                    self.call_log.append(call_stats(data, (time.perf_counter() - start) * 1000, attempt))
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == NETWORK_RETRIES:
                    raise LLMUnavailableError(f"llama-server at {self._url} failed: {e}") from e
                time.sleep(2**attempt)
        raise AssertionError("unreachable")
