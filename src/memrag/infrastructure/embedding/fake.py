"""Deterministic EmbeddingProvider for tests: same encoded text, same vector."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind


class FakeEmbeddingProvider:
    def __init__(self, spec: ModelSpec):
        self._spec = spec

    def info(self) -> ModelSpec:
        return self._spec

    def explain(self, text: str, kind: TextKind) -> EncodedText:
        return self._spec.encode_text(text, kind)

    def embed_queries(self, texts: Sequence[str]):
        return self._embed(texts, "query")

    def embed_documents(self, texts: Sequence[str]):
        return self._embed(texts, "document")

    def _embed(self, texts: Sequence[str], kind: TextKind):
        rows = []
        for t in texts:
            seed = hashlib.sha256(self._spec.encode_text(t, kind).encoded_text.encode("utf-8")).digest()
            v = np.random.default_rng(int.from_bytes(seed[:8], "little")).standard_normal(self._spec.dim)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), self._spec.dim)
