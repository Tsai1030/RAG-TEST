"""Read configs/models.yaml and find downloaded model snapshots."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from memrag.domain.errors import ConfigError, ModelNotAvailableError
from memrag.domain.schemas.embedding import ROLES, ModelSpec

_OPTIONAL = {"download_ignore"}
_REQUIRED = [f.name for f in fields(ModelSpec) if f.name not in _OPTIONAL | {"name"}]
_DTYPES = {"float32", "float16", "bfloat16"}
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class Registry:
    hf_home: Path
    download_ignore: tuple[str, ...]
    models: dict[str, ModelSpec]

    def get(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise ModelNotAvailableError(
                f"model {name!r} is not in configs/models.yaml; known: {', '.join(self.models)}"
            ) from None

    def select(self, names: str) -> list[ModelSpec]:
        """`all`, `candidate`, `baseline`, or a comma-separated list of names."""
        if names in ("all", *ROLES):
            return [s for s in self.models.values() if names == "all" or s.role == names]
        return [self.get(n.strip()) for n in names.split(",") if n.strip()]

    def local_path(self, spec: ModelSpec) -> Path | None:
        """Snapshot folder of the pinned revision, or None if not fully downloaded."""
        folder = self.hf_home / "hub" / f"models--{spec.hf_id.replace('/', '--')}" / "snapshots" / spec.revision
        if not (folder / "modules.json").is_file():
            return None
        if not any((folder / w).is_file() for w in _WEIGHT_FILES):
            return None
        return folder


def load_registry(path: Path, project_root: Path) -> Registry:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"model registry not found: {path}") from None
    settings = raw.get("settings") or {}
    models = raw.get("models") or {}
    if not models:
        raise ConfigError(f"{path}: no models defined")
    specs = {name: _parse_spec(path, name, entry or {}) for name, entry in models.items()}
    return Registry(
        hf_home=(project_root / settings.get("hf_home", ".hf_cache")).resolve(),
        download_ignore=tuple(settings.get("download_ignore") or ()),
        models=specs,
    )


def _parse_spec(path: Path, name: str, entry: dict) -> ModelSpec:
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise ConfigError(f"{path}: model {name!r} is missing {', '.join(missing)}")
    unknown = set(entry) - set(_REQUIRED) - _OPTIONAL
    if unknown:
        raise ConfigError(f"{path}: model {name!r} has unknown fields {', '.join(sorted(unknown))}")
    if entry["role"] not in ROLES:
        raise ConfigError(f"{path}: model {name!r} role must be one of {ROLES}")
    if entry["dtype"] not in _DTYPES:
        raise ConfigError(f"{path}: model {name!r} dtype must be one of {sorted(_DTYPES)}")
    if len(str(entry["revision"])) != 40:
        raise ConfigError(f"{path}: model {name!r} revision must be a full commit hash")
    for key in ("query_prefix", "document_prefix"):
        if not isinstance(entry[key], str):
            raise ConfigError(f"{path}: model {name!r} {key} must be a string (use \"\" for none)")
    return ModelSpec(
        name=name,
        hf_id=entry["hf_id"],
        revision=str(entry["revision"]),
        role=entry["role"],
        dim=int(entry["dim"]),
        max_seq_length=int(entry["max_seq_length"]),
        query_prefix=entry["query_prefix"],
        document_prefix=entry["document_prefix"],
        dtype=entry["dtype"],
        trust_remote_code=bool(entry["trust_remote_code"]),
        license=entry["license"],
        commercial_ok=bool(entry["commercial_ok"]),
        download_ignore=tuple(entry.get("download_ignore") or ()),
    )
