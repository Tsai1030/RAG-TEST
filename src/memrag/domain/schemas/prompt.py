"""A versioned prompt read from prompts/<name>.v<version>.yaml."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from memrag.domain.errors import ConfigError

_VAR = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    status: str
    variables: tuple[str, ...]
    system: str | None
    user: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.system or ''}\n---\n{self.user}".encode("utf-8")).hexdigest()

    def render(self, **values: str) -> tuple[str | None, str]:
        """(system, user) with every {{variable}} filled; missing or unknown variables are errors."""
        missing = set(self.variables) - values.keys()
        extra = values.keys() - set(self.variables)
        if missing or extra:
            raise ConfigError(f"prompt {self.name}.{self.version}: missing {sorted(missing)}, unknown {sorted(extra)}")

        def fill(text: str) -> str:
            return _VAR.sub(lambda m: str(values[m.group(1)]), text)

        return (fill(self.system) if self.system else None), fill(self.user)
