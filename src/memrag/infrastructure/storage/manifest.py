"""Run folders under results/runs/<run_id>/ with a manifest of versions and hardware."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGES = ["torch", "transformers", "sentence-transformers", "chromadb", "huggingface-hub", "numpy"]


def new_run_dir(project_root: Path, kind: str) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{kind}"
    path = project_root / "results" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_info(project_root: Path) -> dict:
    import psutil

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {n: pkg(n) for n in _PACKAGES},
        "git_commit": _git_commit(project_root),
        "mem_available_gib": round(psutil.virtual_memory().available / 2**30, 2),
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
