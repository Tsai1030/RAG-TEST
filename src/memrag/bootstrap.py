"""Wire ports to implementations based on configuration."""

from __future__ import annotations

import os
from pathlib import Path

from memrag.domain.errors import ModelNotAvailableError
from memrag.domain.ports import EmbeddingProvider
from memrag.infrastructure.embedding.registry import Registry, load_registry

MODELS_CONFIG = Path("configs") / "models.yaml"


def project_root() -> Path:
    """MEMRAG_ROOT, or the nearest parent of the current folder holding configs/models.yaml."""
    if env := os.environ.get("MEMRAG_ROOT"):
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / MODELS_CONFIG).is_file():
            return folder
    return here


def registry(root: Path | None = None) -> Registry:
    root = root or project_root()
    return load_registry(root / MODELS_CONFIG, root)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def embedding_provider(reg: Registry, name: str, device: str = "auto", fake: bool = False) -> EmbeddingProvider:
    spec = reg.get(name)
    if fake:
        from memrag.infrastructure.embedding.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider(spec)
    path = reg.local_path(spec)
    if path is None:
        raise ModelNotAvailableError(
            f"model {name!r} ({spec.hf_id}@{spec.revision[:8]}) is not downloaded; run `memrag models download --models {name}`"
        )
    from memrag.infrastructure.embedding.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(spec, path, device=resolve_device(device))
