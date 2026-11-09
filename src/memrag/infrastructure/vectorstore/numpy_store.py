"""VectorStore by brute-force cosine on one contiguous matrix (plan 5.1.7 candidate).

Same results as ExactStore, built for speed: vectors live in one float32 matrix and the
filter fields in integer columns, so a filter is a vectorized mask instead of a Python
loop over records. Records are kept as JSON, like ChromaDB keeps them in metadata.
In memory only; persistence would be needed before using it in the product.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import Category, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter, SearchHit

_CATEGORY_CODE = {c: i for i, c in enumerate(Category)}


def _ord(d) -> int:
    return int(d.strftime("%Y%m%d")) if d else 0


class NumpyStore:
    def __init__(self, dim: int):
        self._dim = dim
        self._row: dict[str, int] = {}
        self._ids: list[str] = []
        self._json: list[str] = []
        self._person_code: dict[str, int] = {}
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._person = np.empty(0, dtype=np.int32)
        self._category = np.empty(0, dtype=np.int8)
        self._event = np.empty(0, dtype=np.int32)
        self._told = np.empty(0, dtype=np.int32)

    def upsert(self, records: Sequence[MemoryRecord], vectors) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(records) != len(vectors) or (len(records) and vectors.shape[1] != self._dim):
            raise ConfigError(f"expected {len(records)} vectors of dim {self._dim}, got {vectors.shape}")
        new = [i for i, r in enumerate(records) if r.memory_id not in self._row]
        start = len(self._ids)
        self._vectors = np.concatenate([self._vectors, np.zeros((len(new), self._dim), dtype=np.float32)])
        self._person = np.concatenate([self._person, np.zeros(len(new), dtype=np.int32)])
        self._category = np.concatenate([self._category, np.zeros(len(new), dtype=np.int8)])
        self._event = np.concatenate([self._event, np.zeros(len(new), dtype=np.int32)])
        self._told = np.concatenate([self._told, np.zeros(len(new), dtype=np.int32)])
        for offset, i in enumerate(new):
            self._row[records[i].memory_id] = start + offset
            self._ids.append(records[i].memory_id)
            self._json.append("")
        for r, v in zip(records, vectors):
            row = self._row[r.memory_id]
            self._vectors[row] = v
            self._person[row] = self._person_code.setdefault(r.person_id, len(self._person_code))
            self._category[row] = _CATEGORY_CODE[r.category]
            self._event[row] = _ord(r.event_date)
            self._told[row] = _ord(r.conversation_date)
            self._json[row] = r.model_dump_json()

    def _mask(self, flt: MemoryFilter | None) -> np.ndarray | None:
        """Rows `flt.matches` accepts, or None for all rows."""
        if flt is None:
            return None
        mask = np.ones(len(self._ids), dtype=bool)
        if flt.person_id is not None:
            code = self._person_code.get(flt.person_id)
            mask &= self._person == code if code is not None else False
        if flt.category is not None:
            mask &= self._category == _CATEGORY_CODE[flt.category]
        if flt.categories is not None:
            mask &= np.isin(self._category, [_CATEGORY_CODE[c] for c in flt.categories])
        if flt.has_date:
            day = np.where(self._event > 0, self._event, self._told if flt.date_fallback else 0)
            mask &= day > 0
            if flt.date_from:
                mask &= day >= _ord(flt.date_from)
            if flt.date_to:
                mask &= day <= _ord(flt.date_to)
        return mask

    def matching_ids(self, flt: MemoryFilter) -> set[str]:
        mask = self._mask(flt)
        rows = range(len(self._ids)) if mask is None else np.flatnonzero(mask)
        return {self._ids[i] for i in rows}

    def match_count(self, flt: MemoryFilter | None) -> int:
        mask = self._mask(flt)
        return len(self._ids) if mask is None else int(mask.sum())

    def search(self, vector, k: int, flt: MemoryFilter | None = None) -> list[SearchHit]:
        if k <= 0 or not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        mask = self._mask(flt)
        rows = np.arange(len(self._ids)) if mask is None else np.flatnonzero(mask)
        if not len(rows):
            return []
        scores = (self._vectors if mask is None else self._vectors[rows]) @ q
        if len(rows) > k:
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
        else:
            top = np.argsort(-scores, kind="stable")
        return [SearchHit(self._ids[rows[j]], float(scores[j])) for j in top]

    def get(self, ids: Sequence[str]) -> list[MemoryRecord]:
        return [MemoryRecord.model_validate_json(self._json[self._row[i]]) for i in ids if i in self._row]

    def count(self) -> int:
        return len(self._ids)

    def nbytes(self) -> int:
        """Memory held by the index: vectors, filter columns and record JSON."""
        arrays = (self._vectors, self._person, self._category, self._event, self._told)
        return sum(a.nbytes for a in arrays) + sum(len(s.encode("utf-8")) for s in self._json)
