"""VectorStore by brute-force cosine in memory: the reference for filters and recall."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter, SearchHit


class ExactStore:
    def __init__(self, dim: int):
        self._dim = dim
        self._records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, records: Sequence[MemoryRecord], vectors) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(records) != len(vectors) or (len(records) and vectors.shape[1] != self._dim):
            raise ConfigError(f"expected {len(records)} vectors of dim {self._dim}, got {vectors.shape}")
        for r, v in zip(records, vectors):
            self._records[r.memory_id] = r
            self._vectors[r.memory_id] = v

    def get(self, ids: Sequence[str]) -> list[MemoryRecord]:
        return [self._records[i] for i in ids if i in self._records]

    def matching_ids(self, flt: MemoryFilter) -> set[str]:
        return {i for i, r in self._records.items() if flt.matches(r)}

    def search(self, vector, k: int, flt: MemoryFilter | None = None) -> list[SearchHit]:
        ids = [i for i, r in self._records.items() if flt is None or flt.matches(r)]
        if not ids or k <= 0:
            return []
        scores = np.stack([self._vectors[i] for i in ids]) @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(ids[j], float(scores[j])) for j in order]

    def count(self) -> int:
        return len(self._records)
