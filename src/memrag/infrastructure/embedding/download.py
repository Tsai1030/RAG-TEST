"""Download pinned model revisions into the project's own Hugging Face cache."""

from __future__ import annotations

from pathlib import Path

from memrag.domain.schemas.embedding import ModelSpec
from memrag.infrastructure.embedding.registry import Registry


def download_model(registry: Registry, spec: ModelSpec) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        spec.hf_id,
        revision=spec.revision,
        cache_dir=registry.hf_home / "hub",
        ignore_patterns=[*registry.download_ignore, *spec.download_ignore],
    )
    return Path(path)


def snapshot_size(path: Path) -> int:
    """Bytes of the files the snapshot points to (snapshots hold symlinks)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
