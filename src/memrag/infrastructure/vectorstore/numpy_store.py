"""VectorStore by brute-force cosine on one contiguous matrix (the product store since 5.1.7).

Same results as ExactStore, built for speed: vectors live in one float32 matrix and the
filter fields in integer columns, so a filter is a vectorized mask instead of a Python
loop over records. Records are kept as JSON, like ChromaDB keeps them in metadata.

With a path, each model and dimension gets a folder (named like the ChromaDB collection):
`meta.json` (model, dim), `vectors.f32` (raw float32 rows) and `records.jsonl` (one record
per row, same order). New memories are appended; a changed memory rewrites both files
through a temporary file. On open, rows present in only one file (a write cut off between
the two appends) are dropped. One writer at a time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.memory import Category, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter, SearchHit
from memrag.infrastructure.vectorstore.chroma_store import collection_name

_CATEGORY_CODE = {c: i for i, c in enumerate(Category)}
FORMAT = 1


def _ord(d) -> int:
    return int(d.strftime("%Y%m%d")) if d else 0


class NumpyStore:
    def __init__(self, dim: int, path: Path | None = None, model: str | None = None):
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
        self._dir = None
        if path is not None:
            if not model:
                raise ConfigError("a persistent NumpyStore needs the model name")
            self._dir = Path(path) / collection_name(model, dim)
            self._open(model)

    # --- persistence -------------------------------------------------------------------

    def _open(self, model: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._dir / "meta.json"
        meta = {"model": model, "dim": self._dim, "format": FORMAT}
        if not meta_path.is_file():
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            return
        found = json.loads(meta_path.read_text(encoding="utf-8"))
        if found != meta:
            raise ConfigError(f"store {self._dir} was created for {found}, not {meta}")
        lines = (self._dir / "records.jsonl").read_text(encoding="utf-8").splitlines() \
            if (self._dir / "records.jsonl").is_file() else []
        raw = np.fromfile(self._dir / "vectors.f32", dtype=np.float32) \
            if (self._dir / "vectors.f32").is_file() else np.empty(0, dtype=np.float32)
        n = min(len(lines), raw.size // self._dim)
        if n != len(lines) or n * self._dim != raw.size:  # cut off between the two appends
            lines, raw = lines[:n], raw[:n * self._dim]
            self._rewrite_files(lines, raw.reshape(n, self._dim))
        self._load_rows([MemoryRecord.model_validate_json(line) for line in lines], raw.reshape(n, self._dim))

    def _rewrite_files(self, lines: list[str], vectors: np.ndarray) -> None:
        for name, write in (("vectors.f32", lambda f: f.write(np.ascontiguousarray(vectors, np.float32).tobytes())),
                            ("records.jsonl", lambda f: f.write("".join(s + "\n" for s in lines).encode("utf-8")))):
            tmp = self._dir / (name + ".tmp")
            with tmp.open("wb") as f:
                write(f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._dir / name)

    def _append_files(self, rows: list[int]) -> None:
        with (self._dir / "vectors.f32").open("ab") as f:
            f.write(np.ascontiguousarray(self._vectors[rows]).tobytes())
            f.flush()
            os.fsync(f.fileno())
        with (self._dir / "records.jsonl").open("ab") as f:
            f.write("".join(self._json[r] + "\n" for r in rows).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    # --- writes --------------------------------------------------------------------------

    def _load_rows(self, records: Sequence[MemoryRecord], vectors: np.ndarray) -> tuple[list[int], bool]:
        """Put records into memory; returns the new rows and whether an existing row changed."""
        new, seen = [], set()
        for i, r in enumerate(records):  # an id twice in one batch: the last one wins, like ExactStore
            if r.memory_id not in self._row and r.memory_id not in seen:
                new.append(i)
                seen.add(r.memory_id)
        start = len(self._ids)
        grow = len(new)
        self._vectors = np.concatenate([self._vectors, np.zeros((grow, self._dim), dtype=np.float32)])
        self._person = np.concatenate([self._person, np.zeros(grow, dtype=np.int32)])
        self._category = np.concatenate([self._category, np.zeros(grow, dtype=np.int8)])
        self._event = np.concatenate([self._event, np.zeros(grow, dtype=np.int32)])
        self._told = np.concatenate([self._told, np.zeros(grow, dtype=np.int32)])
        new_rows = []
        for offset, i in enumerate(new):
            self._row[records[i].memory_id] = start + offset
            self._ids.append(records[i].memory_id)
            self._json.append("")
            new_rows.append(start + offset)
        changed = False
        new_set = set(new_rows)
        for r, v in zip(records, vectors):
            row = self._row[r.memory_id]
            text = r.model_dump_json()
            if row not in new_set:
                if text == self._json[row] and np.array_equal(self._vectors[row], v):
                    continue
                changed = True
            self._vectors[row] = v
            self._person[row] = self._person_code.setdefault(r.person_id, len(self._person_code))
            self._category[row] = _CATEGORY_CODE[r.category]
            self._event[row] = _ord(r.event_date)
            self._told[row] = _ord(r.conversation_date)
            self._json[row] = text
        return new_rows, changed

    def upsert(self, records: Sequence[MemoryRecord], vectors) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(records) != len(vectors) or (len(records) and vectors.shape[1] != self._dim):
            raise ConfigError(f"expected {len(records)} vectors of dim {self._dim}, got {vectors.shape}")
        new_rows, changed = self._load_rows(records, vectors)
        if self._dir is None:
            return
        if changed:
            self._rewrite_files(self._json, self._vectors)
        elif new_rows:
            self._append_files(new_rows)

    # --- reads ---------------------------------------------------------------------------

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

    def close(self) -> None:
        """Nothing to release: every write is already on disk."""
