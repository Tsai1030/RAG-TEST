"""Load one model in a fresh subprocess and measure load time, memory and latency.

Each (model, device) runs in its own process so memory peaks do not leak into
the next measurement. The parent passes the spec as JSON and reads one JSON
line back from stdout.

Memory on the Jetson is shared by CPU and GPU, so besides the PyTorch numbers
we sample system MemAvailable every 20 ms and report the largest drop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from memrag.domain.schemas.embedding import ModelSpec
from memrag.infrastructure.memory_sampler import MemSampler as _MemSampler
from memrag.infrastructure.memory_sampler import mib as _mib

LATENCY_QUERY = "田中さんの好きな食べ物は何ですか？"
BATCH_DOCUMENT = "田中さんは先週、駅前の喫茶店で友達と会って、来月の旅行の計画を話した。"
LONG_SENTENCE = "今日は朝から雨が降っていたので、駅前の喫茶店で友達と長い時間話をした。"
WARMUP_RUNS = 3
LATENCY_RUNS = 20
BATCH_SIZE = 32


def run_probe(spec: ModelSpec, model_path: Path, device: str, timeout_s: int = 900) -> dict:
    """Run the measurement in a child process and return its result."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    cmd = [sys.executable, "-m", "memrag.infrastructure.embedding.probe",
           json.dumps(asdict(spec), ensure_ascii=False), str(model_path), device]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return _failure(spec, device, f"timeout after {timeout_s}s", time.time() - started)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        reason = "killed by signal (likely out of memory)" if proc.returncode < 0 else f"exit code {proc.returncode}"
        return _failure(spec, device, reason, time.time() - started, tail)
    return json.loads(lines[-1])


def _failure(spec: ModelSpec, device: str, reason: str, elapsed: float, log: list[str] | None = None) -> dict:
    return {"model": spec.name, "device": device, "ok": False, "error": reason,
            "elapsed_s": round(elapsed, 1), "log_tail": log or []}


def _measure(spec: ModelSpec, model_path: Path, device: str) -> dict:
    import resource

    import psutil

    sampler = _MemSampler()
    sampler.start()
    t0 = time.perf_counter()
    import numpy as np
    import torch

    from memrag.infrastructure.embedding.sentence_transformer import SentenceTransformerProvider

    import_s = time.perf_counter() - t0
    cuda = device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    t1 = time.perf_counter()
    provider = SentenceTransformerProvider(spec, model_path, device=device, batch_size=BATCH_SIZE)
    sync()
    load_s = time.perf_counter() - t1
    model = provider.model
    after_load_mem = sampler.baseline - psutil.virtual_memory().available
    after_load_cuda = torch.cuda.memory_allocated() if cuda else None

    for _ in range(WARMUP_RUNS):
        provider.embed_queries([LATENCY_QUERY])
    sync()
    latencies = []
    for _ in range(LATENCY_RUNS):
        t = time.perf_counter()
        vec = provider.embed_queries([LATENCY_QUERY])
        sync()
        latencies.append((time.perf_counter() - t) * 1000)
    latencies.sort()

    docs = [f"{BATCH_DOCUMENT}（{i}）" for i in range(BATCH_SIZE)]
    t = time.perf_counter()
    provider.embed_documents(docs)
    sync()
    batch_ms = (time.perf_counter() - t) * 1000

    long_text = LONG_SENTENCE * 40
    long_tokens = len(model.tokenizer(spec.document_prefix + long_text)["input_ids"])
    t = time.perf_counter()
    long_vec = provider.embed_documents([long_text])
    sync()
    long_ms = (time.perf_counter() - t) * 1000

    peak_sys = sampler.stop()
    first_param = next(model.parameters())
    return {
        "model": spec.name,
        "device": device,
        "ok": True,
        "dtype": str(first_param.dtype).replace("torch.", ""),
        "torch_threads": torch.get_num_threads(),
        "import_s": round(import_s, 2),
        "load_s": round(load_s, 2),
        "dim": int(vec.shape[1]),
        "dim_matches_config": int(vec.shape[1]) == spec.dim,
        "norm": round(float(np.linalg.norm(vec[0])), 4),
        "latency_ms_p50": round(latencies[len(latencies) // 2], 1),
        "latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "latency_ms_min": round(latencies[0], 1),
        "batch32_ms": round(batch_ms, 1),
        "long_input_chars": len(long_text),
        "long_input_tokens": long_tokens,
        "max_seq_length": model.max_seq_length,
        "long_input_truncated": long_tokens > model.max_seq_length,
        "long_input_ms": round(long_ms, 1),
        "long_input_finite": bool(np.isfinite(long_vec).all()),
        "sys_mem_after_load_mib": _mib(after_load_mem),
        "sys_mem_peak_mib": _mib(peak_sys),
        "process_peak_rss_mib": _mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "cuda_after_load_mib": _mib(after_load_cuda) if cuda else None,
        "cuda_peak_mib": _mib(torch.cuda.max_memory_allocated()) if cuda else None,
    }


def main(argv: list[str]) -> int:
    spec_json, model_path, device = argv
    data = json.loads(spec_json)
    data["download_ignore"] = tuple(data.get("download_ignore") or ())
    result = _measure(ModelSpec(**data), Path(model_path), device)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
