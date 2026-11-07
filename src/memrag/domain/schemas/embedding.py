"""Embedding model specification and prefix rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TextKind = Literal["query", "document"]
TEXT_KINDS: tuple[TextKind, ...] = ("query", "document")
ROLES = ("candidate", "baseline")


@dataclass(frozen=True)
class ModelSpec:
    """One entry of configs/models.yaml."""

    name: str
    hf_id: str
    revision: str
    role: str
    dim: int
    max_seq_length: int
    query_prefix: str
    document_prefix: str
    dtype: str
    trust_remote_code: bool
    license: str
    commercial_ok: bool
    download_ignore: tuple[str, ...] = ()

    def prefix(self, kind: TextKind) -> str:
        if kind == "query":
            return self.query_prefix
        if kind == "document":
            return self.document_prefix
        raise ValueError(f"kind must be one of {TEXT_KINDS}, got {kind!r}")

    def encode_text(self, text: str, kind: TextKind) -> EncodedText:
        prefix = self.prefix(kind)
        return EncodedText(model=self.name, kind=kind, text=text, prefix=prefix, encoded_text=prefix + text)


@dataclass(frozen=True)
class EncodedText:
    """The exact text sent to the model, for debugging prefixes."""

    model: str
    kind: TextKind
    text: str
    prefix: str
    encoded_text: str
