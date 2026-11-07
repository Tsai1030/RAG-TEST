"""Port interfaces. Application code depends on these, never on implementations.

Other ports in the plan (LLMProvider, PromptStore, VectorStore, Profiler) are
added with the tasks that implement them.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind

Vector = Sequence[float]


class EmbeddingProvider(Protocol):
    """Turns text into L2-normalized vectors, applying the model's prefix."""

    def info(self) -> ModelSpec: ...

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def explain(self, text: str, kind: TextKind) -> EncodedText: ...
