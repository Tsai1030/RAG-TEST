"""Wire ports to implementations based on configuration."""

from __future__ import annotations

import os
from pathlib import Path

from memrag.domain.errors import ModelNotAvailableError
from memrag.domain.ports import EmbeddingProvider
from memrag.infrastructure.embedding.registry import Registry, load_registry

MODELS_CONFIG = Path("configs") / "models.yaml"
GOLD_DIR = Path("data") / "00_gold"
GOLD_FILES = {
    "personas": "personas.jsonl",
    "facts": "gold_facts.jsonl",
    "queries": "queries.jsonl",
    "labels": "labels.jsonl",
    "labels_second": "labels_second.jsonl",
    "second_sample": "second_pass_sample.json",
}


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


def gold_path(name: str, root: Path | None = None) -> Path:
    return (root or project_root()) / GOLD_DIR / GOLD_FILES[name]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def release_accelerator_memory() -> None:
    """Free cached GPU memory after a model is dropped, so the next model starts clean."""
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def embedding_provider(reg: Registry, name: str, device: str = "auto", fake: bool = False,
                       dtype: str | None = None) -> EmbeddingProvider:
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

    return SentenceTransformerProvider(spec, path, device=resolve_device(device), dtype=dtype)
