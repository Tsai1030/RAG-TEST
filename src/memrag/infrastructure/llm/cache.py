"""ResponseCache as one small JSON file per key."""

from __future__ import annotations

import json
from pathlib import Path


class FileResponseCache:
    def __init__(self, folder: Path):
        self._folder = folder

    def _path(self, key: str) -> Path:
        return self._folder / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        return json.loads(path.read_text(encoding="utf-8"))["output"] if path.is_file() else None

    def put(self, key: str, value: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "output": value}, ensure_ascii=False) + "\n", encoding="utf-8")
