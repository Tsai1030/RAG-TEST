"""Port interfaces. Application code depends on these, never on implementations.

The Profiler port in the plan is added with the task that implements it.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind
from memrag.domain.schemas.memory import MemoryRecord
from memrag.domain.schemas.prompt import PromptTemplate
from memrag.domain.schemas.retrieval import MemoryFilter, SearchHit

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


class VectorStore(Protocol):
    """Memories and their L2-normalized vectors for one embedding model.

    `upsert` is idempotent (memory_id is the key). `search` returns at most k hits,
    best first, all accepted by `flt.matches`; fewer when fewer memories match.
    """

    def upsert(self, records: Sequence[MemoryRecord], vectors: Sequence[Vector]) -> None: ...

    def search(self, vector: Vector, k: int, flt: MemoryFilter | None = None) -> list[SearchHit]: ...

    def get(self, ids: Sequence[str]) -> list[MemoryRecord]:
        """The stored records, in the order of `ids`; unknown ids are skipped."""
        ...

    def matching_ids(self, flt: MemoryFilter) -> set[str]:
        """Every stored memory the filter accepts, without ranking (to check filter translation)."""
        ...

    def count(self) -> int: ...


class PromptStore(Protocol):
    def get(self, name: str, version: str) -> PromptTemplate: ...


class ResponseCache(Protocol):
    """Valid LLM outputs keyed by a hash of everything that affects them."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, value: str) -> None: ...
