"""Facade: the stable public API. CLI, UI and scripts only call this module."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from memrag import bootstrap
from memrag.domain.ports import EmbeddingProvider
from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind


class MemRAG:
    def __init__(self, embedder: EmbeddingProvider):
        self._embedder = embedder

    @classmethod
    def from_config(cls, embedding_model: str, device: str = "auto", fake: bool = False) -> MemRAG:
        reg = bootstrap.registry()
        return cls(bootstrap.embedding_provider(reg, embedding_model, device=device, fake=fake))

    @property
    def embedding_model(self) -> ModelSpec:
        return self._embedder.info()

    def embed(self, texts: Sequence[str], kind: TextKind = "query"):
        if kind == "query":
            return self._embedder.embed_queries(texts)
        return self._embedder.embed_documents(texts)

    def explain_embedding(self, text: str, kind: TextKind = "query") -> EncodedText:
        return self._embedder.explain(text, kind)


def list_models(selection: str = "all") -> list[dict]:
    reg = bootstrap.registry()
    rows = []
    for spec in reg.select(selection):
        path = reg.local_path(spec)
        rows.append({**asdict(spec), "downloaded": path is not None, "local_path": str(path) if path else None})
    return rows


def check_environment(require_gpu: bool | None = None) -> dict:
    """Run every environment check. `require_gpu=None` requires a GPU only on Linux aarch64 (the Nano)."""
    import platform

    from memrag.infrastructure.environment import run_checks

    if require_gpu is None:
        require_gpu = platform.system() == "Linux" and platform.machine() == "aarch64"
    root = bootstrap.project_root()
    checks = run_checks(bootstrap.registry(root), root, require_gpu)
    return {"ok": all(c["ok"] for c in checks if c["name"] != "models"), "checks": checks}


def download_models(selection: str = "all") -> list[dict]:
    from memrag.infrastructure.embedding.download import download_model, snapshot_size

    reg = bootstrap.registry()
    rows = []
    for spec in reg.select(selection):
        path = download_model(reg, spec)
        rows.append({"model": spec.name, "revision": spec.revision, "path": str(path),
                     "size_mib": round(snapshot_size(path) / 2**20, 1)})
    return rows


def probe_models(selection: str = "all", devices: Sequence[str] = ("cuda", "cpu"), note: str = "") -> dict:
    """Measure loading and latency of each model on each device, one subprocess per pair.

    Results go to results/runs/<run_id>/load_probe.jsonl with a manifest.json.
    """
    from memrag.infrastructure.embedding.probe import run_probe
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    root = bootstrap.project_root()
    reg = bootstrap.registry(root)
    specs = reg.select(selection)
    run_dir = new_run_dir(root, "load_probe")
    manifest = {
        "run_id": run_dir.name,
        "kind": "load_probe",
        "started": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "devices": list(devices),
        "models": {s.name: {"hf_id": s.hf_id, "revision": s.revision, "dtype": s.dtype} for s in specs},
        "environment": environment_info(root),
    }
    write_json(run_dir / "manifest.json", manifest)
    results = []
    for spec in specs:
        path = reg.local_path(spec)
        for device in devices:
            if path is None:
                row = {"model": spec.name, "device": device, "ok": False, "error": "not downloaded"}
            else:
                row = run_probe(spec, path, device)
            append_jsonl(run_dir / "load_probe.jsonl", row)
            results.append(row)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "results": results}
