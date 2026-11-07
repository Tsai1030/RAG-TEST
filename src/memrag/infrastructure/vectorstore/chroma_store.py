"""VectorStore on ChromaDB in embedded mode (HNSW, cosine).

One collection per model and dimension (`mem__<model>__<dim>`), so vectors of different
models can never be mixed. Chroma metadata cannot hold None, so a memory without an
event date stores event_date_ord = 0; any date bound adds `event_date_ord > 0`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter, SearchHit

UPSERT_BATCH = 500


def collection_name(model: str, dim: int) -> str:
    return f"mem__{re.sub(r'[^a-zA-Z0-9_-]', '_', model)}__{dim}"


def to_where(flt: MemoryFilter | None) -> dict | None:
    """MemoryFilter -> Chroma `where`; the only place that knows Chroma's filter syntax."""
    if flt is None:
        return None
    terms: list[dict] = []
    if flt.person_id is not None:
        terms.append({"person_id": {"$eq": flt.person_id}})
    if flt.category is not None:
        terms.append({"category": {"$eq": flt.category.value}})
    if flt.has_date:
        terms.append({"event_date_ord": {"$gt": 0}})
        if flt.date_from:
            terms.append({"event_date_ord": {"$gte": int(flt.date_from.strftime("%Y%m%d"))}})
        if flt.date_to:
            terms.append({"event_date_ord": {"$lte": int(flt.date_to.strftime("%Y%m%d"))}})
    if not terms:
        return None
    return terms[0] if len(terms) == 1 else {"$and": terms}


class ChromaStore:
    def __init__(self, model: str, dim: int, path: Path | None = None):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        name = collection_name(model, dim)
        self._col = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine", "model": model, "dim": dim})
        meta = self._col.metadata or {}
        if meta.get("model") != model or meta.get("dim") != dim or meta.get("hnsw:space") != "cosine":
            raise ConfigError(f"collection {name} was created for {meta}, not model={model} dim={dim} cosine")
        self._dim = dim

    def upsert(self, records: Sequence[MemoryRecord], vectors) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(records) != len(vectors) or (len(records) and vectors.shape[1] != self._dim):
            raise ConfigError(f"expected {len(records)} vectors of dim {self._dim}, got {vectors.shape}")
        for start in range(0, len(records), UPSERT_BATCH):
            batch = records[start:start + UPSERT_BATCH]
            self._col.upsert(
                ids=[r.memory_id for r in batch],
                embeddings=vectors[start:start + UPSERT_BATCH].tolist(),
                documents=[r.content for r in batch],
                metadatas=[{"person_id": r.person_id, "category": r.category.value,
                            "event_date_ord": r.event_date_ord or 0,
                            "conversation_date_ord": int(r.conversation_date.strftime("%Y%m%d"))} for r in batch],
            )

    def matching_ids(self, flt: MemoryFilter) -> set[str]:
        return set(self._col.get(where=to_where(flt), include=[])["ids"])

    def search(self, vector, k: int, flt: MemoryFilter | None = None) -> list[SearchHit]:
        if k <= 0 or self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                              n_results=min(k, self._col.count()), where=to_where(flt), include=["distances"])
        return [SearchHit(i, 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]

    def count(self) -> int:
        return self._col.count()

    def close(self) -> None:
        """Release the files (Windows keeps them open otherwise)."""
        import chromadb

        del self._col
        self._client = None
        chromadb.api.client.SharedSystemClient.clear_system_cache()
