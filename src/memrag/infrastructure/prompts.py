"""PromptStore reading prompts/<name>.v<version>.yaml (format in prompts/README.md)."""

from __future__ import annotations

from pathlib import Path

import yaml

from memrag.domain.errors import ConfigError
from memrag.domain.schemas.prompt import PromptTemplate


class YamlPromptStore:
    def __init__(self, folder: Path):
        self._folder = folder

    def get(self, name: str, version: str) -> PromptTemplate:
        path = self._folder / f"{name}.{version}.yaml"
        if not path.is_file():
            raise ConfigError(f"prompt not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("name") != name or data.get("version") != version:
            raise ConfigError(f"{path.name}: name/version fields do not match the file name")
        return PromptTemplate(
            name=name,
            version=version,
            status=data.get("status", "draft"),
            variables=tuple(data.get("variables") or ()),
            system=data.get("system"),
            user=data["user"],
        )
