"""Environment checks behind `memrag env check`.

Each check returns a dict with `name`, `ok` and `detail`; a failed check never
raises, so one problem does not hide the others.
"""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from memrag.infrastructure.embedding.registry import Registry


def _check(name: str, fn) -> dict:
    t = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001 - report every failure, keep checking
        ok, detail = False, f"{type(e).__name__}: {e}"
    return {"name": name, "ok": ok, "detail": detail, "seconds": round(time.perf_counter() - t, 2)}


def check_python() -> tuple[bool, dict]:
    return sys.version_info[:2] == (3, 11), {
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
    }


def check_packages() -> tuple[bool, dict]:
    from importlib.metadata import version

    names = ["torch", "transformers", "sentence-transformers", "chromadb", "huggingface-hub", "numpy"]
    return True, {n: version(n) for n in names}


def check_gpu(require: bool) -> tuple[bool, dict]:
    import torch

    if not torch.cuda.is_available():
        return not require, {"cuda_available": False, "required": require}
    a = torch.randn(512, 512, generator=torch.Generator().manual_seed(0))
    b = torch.randn(512, 512, generator=torch.Generator().manual_seed(1))
    gpu = (a.cuda() @ b.cuda()).cpu()
    max_diff = float((gpu - a @ b).abs().max())
    return max_diff < 1e-2, {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_cuda": torch.version.cuda,
        "matmul_512_max_abs_diff_vs_cpu": round(max_diff, 6),
    }


def check_chromadb() -> tuple[bool, dict]:
    import chromadb

    # Windows keeps the index files open after the client is gone, so cleanup errors are ignored there
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = chromadb.PersistentClient(path=tmp)
        col = client.create_collection("env_check", metadata={"hnsw:space": "cosine"})
        col.add(
            ids=["m1", "m2", "m3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            documents=["田中さんはラーメンが好き", "佐藤さんは犬を飼っている", "田中さんは寿司も好き"],
            metadatas=[{"person_id": "p001"}, {"person_id": "p002"}, {"person_id": "p001"}],
        )
        res = col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=3, where={"person_id": "p001"})
        ids = res["ids"][0]
        count = col.count()
        del col, client
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    ok = ids == ["m1", "m3"] and count == 3
    return ok, {"version": chromadb.__version__, "filtered_query_ids": ids, "count": count}


def check_models(registry: Registry) -> tuple[bool, dict]:
    status = {s.name: registry.local_path(s) is not None for s in registry.models.values()}
    return all(status.values()), {"hf_home": str(registry.hf_home), "downloaded": status}


def check_resources(project_root: Path) -> tuple[bool, dict]:
    import psutil

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(project_root)
    return True, {
        "mem_total_gib": round(vm.total / 2**30, 2),
        "mem_available_gib": round(vm.available / 2**30, 2),
        "disk_free_gib": round(disk.free / 2**30, 1),
    }


def run_checks(registry: Registry, project_root: Path, require_gpu: bool) -> list[dict]:
    return [
        _check("python", check_python),
        _check("packages", check_packages),
        _check("gpu", lambda: check_gpu(require_gpu)),
        _check("chromadb", check_chromadb),
        _check("models", lambda: check_models(registry)),
        _check("resources", lambda: check_resources(project_root)),
    ]
