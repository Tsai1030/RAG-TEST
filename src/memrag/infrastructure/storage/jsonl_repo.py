"""Read and write JSONL files of pydantic models; errors name the file and line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel, ValidationError

from memrag.domain.errors import ConfigError

M = TypeVar("M", bound=BaseModel)


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        raise ConfigError(f"data file not found: {path}")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ConfigError(f"{path.name} line {n}: {e}") from None
    return rows


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r.model_dump(mode="json", exclude_defaults=False), ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
