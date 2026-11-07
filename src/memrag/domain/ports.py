"""Port interfaces. Application code depends on these, never on implementations.

Other ports in the plan (VectorStore, Profiler) are added with the tasks that
implement them.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind
from memrag.domain.schemas.prompt import PromptTemplate

Vector = Sequence[float]
ChatMessage = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


class EmbeddingProvider(Protocol):
    """Turns text into L2-normalized vectors, applying the model's prefix."""

    def info(self) -> ModelSpec: ...

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def explain(self, text: str, kind: TextKind) -> EncodedText: ...


class LLMProvider(Protocol):
    """Chat completion constrained to a JSON schema. Retries network errors itself."""

    def info(self) -> dict:
        """Model and sampling parameters; part of the cache key."""
        ...

    def chat(self, messages: Sequence[ChatMessage], json_schema: dict) -> str: ...


class PromptStore(Protocol):
    def get(self, name: str, version: str) -> PromptTemplate: ...


class ResponseCache(Protocol):
    """Valid LLM outputs keyed by a hash of everything that affects them."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, value: str) -> None: ...
