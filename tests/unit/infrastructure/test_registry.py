from pathlib import Path

import pytest
import yaml

from memrag.domain.errors import ConfigError, ModelNotAvailableError
from memrag.infrastructure.embedding.registry import load_registry

ROOT = Path(__file__).resolve().parents[3]

ENTRY = {
    "hf_id": "org/m", "revision": "a" * 40, "role": "candidate", "dim": 4, "max_seq_length": 512,
    "query_prefix": "query: ", "document_prefix": "passage: ", "dtype": "float32",
    "trust_remote_code": False, "license": "MIT", "commercial_ok": True,
}


def write(tmp_path: Path, models: dict) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"settings": {"hf_home": ".hf"}, "models": models}, allow_unicode=True), encoding="utf-8")
    return path


def test_project_registry_loads():
    reg = load_registry(ROOT / "configs" / "models.yaml", ROOT)
    assert len(reg.select("candidate")) >= 4
    assert [s.name for s in reg.select("baseline")] == ["bge-base-zh-v1.5"]
    assert all(s.commercial_ok for s in reg.select("candidate")), "D3: candidates must allow commercial use"


def test_hf_home_is_inside_project(tmp_path):
    reg = load_registry(write(tmp_path, {"m": ENTRY}), tmp_path)
    assert reg.hf_home == (tmp_path / ".hf").resolve()


@pytest.mark.parametrize("field", ["revision", "query_prefix", "dtype"])
def test_missing_field_is_reported(tmp_path, field):
    entry = {k: v for k, v in ENTRY.items() if k != field}
    with pytest.raises(ConfigError, match=field):
        load_registry(write(tmp_path, {"m": entry}), tmp_path)


def test_short_revision_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="commit hash"):
        load_registry(write(tmp_path, {"m": ENTRY | {"revision": "main"}}), tmp_path)


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown"):
        load_registry(write(tmp_path, {"m": ENTRY | {"prefix": "x"}}), tmp_path)


def test_select_by_names_and_unknown_name(tmp_path):
    reg = load_registry(write(tmp_path, {"a": ENTRY, "b": ENTRY}), tmp_path)
    assert [s.name for s in reg.select("b, a")] == ["b", "a"]
    with pytest.raises(ModelNotAvailableError, match="known: a, b"):
        reg.select("c")


def test_local_path_requires_weights(tmp_path):
    reg = load_registry(write(tmp_path, {"m": ENTRY}), tmp_path)
    spec = reg.get("m")
    snap = reg.hf_home / "hub" / "models--org--m" / "snapshots" / spec.revision
    snap.mkdir(parents=True)
    (snap / "modules.json").write_text("[]")
    assert reg.local_path(spec) is None
    (snap / "model.safetensors").write_bytes(b"")
    assert reg.local_path(spec) == snap
