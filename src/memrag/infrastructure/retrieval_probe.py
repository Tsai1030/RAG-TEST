"""Measure one retrieval configuration in a fresh subprocess (plan 5.1.7).

The parent writes a job file: records (JSONL), optional precomputed vectors (.npy),
queries with one filter per scenario, the store, Top-k values and, when query or
document vectors must be computed, the embedding model and device. The child builds
the store, times every stage of every query and prints one JSON line with raw samples;
the parent summarizes. Each configuration gets its own process, so memory measured
here belongs to this configuration only.

Stages per query: embed (question + prefix → vector; skipped when query vectors are
given), search (filter + Top-k), get (full records). Memory: system MemAvailable drop
(sampled every 20 ms), process peak RSS, CUDA peak.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from memrag.infrastructure.memory_sampler import MemSampler, mib, peak_rss


def run_job(job: dict, job_dir: Path, timeout_s: int = 3600) -> dict:
    """Run one job in a child process and return its result (or the failure)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "job.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "ANONYMIZED_TELEMETRY": "False"}
    cmd = [sys.executable, "-m", "memrag.infrastructure.retrieval_probe", str(job_file)]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s", "elapsed_s": round(time.time() - started, 1)}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        reason = "killed by signal (likely out of memory)" if proc.returncode < 0 else f"exit code {proc.returncode}"
        return {"ok": False, "error": reason, "elapsed_s": round(time.time() - started, 1), "log_tail": tail}
    return json.loads(lines[-1])


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _embedder(cfg: dict):
    from memrag.domain.schemas.embedding import ModelSpec

    data = dict(cfg["spec"])
    data["download_ignore"] = tuple(data.get("download_ignore") or ())
    spec = ModelSpec(**data)
    if cfg.get("fake"):
        from memrag.infrastructure.embedding.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider(spec)
    from memrag.infrastructure.embedding.sentence_transformer import SentenceTransformerProvider

    return SentenceTransformerProvider(spec, Path(cfg["path"]), device=cfg["device"])


def _measure(job: dict) -> dict:
    import numpy as np

    from memrag.domain.schemas.memory import MemoryRecord
    from memrag.domain.schemas.retrieval import MemoryFilter

    sampler = MemSampler()
    sampler.start()
    emb_cfg = job.get("embedding")
    cuda = bool(emb_cfg) and str(emb_cfg["device"]).startswith("cuda") and not emb_cfg.get("fake")
    torch = None
    if cuda:
        import torch

        if not torch.cuda.is_available():
            sampler.stop()
            return {"ok": False, "error": "CUDA is not available"}

    def sync():
        if torch is not None:
            torch.cuda.synchronize()

    records = [MemoryRecord.model_validate_json(line)
               for line in Path(job["records"]).read_text(encoding="utf-8").splitlines() if line]
    out: dict = {"ok": True, "store": job["store"], "records": len(records),
                 "device": emb_cfg["device"] if emb_cfg else None}

    embedder = None
    if emb_cfg:
        t = time.perf_counter()
        embedder = _embedder(emb_cfg)
        sync()
        out["model_load_s"] = round(time.perf_counter() - t, 2)
        out["mem_after_model_load_mib"] = mib(sampler.used_now())

    if job.get("vectors"):
        vectors = np.load(job["vectors"])
        out["embed_documents_s"] = None
    else:
        t = time.perf_counter()
        vectors = np.asarray(embedder.embed_documents([r.content for r in records]), dtype=np.float32)
        sync()
        out["embed_documents_s"] = round(time.perf_counter() - t, 3)

    mem_before_store = sampler.used_now()
    db_dir = Path(job["work_dir"]) / "db"
    if job["store"] == "chroma":
        from memrag.infrastructure.vectorstore.chroma_store import ChromaStore

        shutil.rmtree(db_dir, ignore_errors=True)
        store = ChromaStore(job["model"], int(vectors.shape[1]), db_dir)
    else:
        from memrag.infrastructure.vectorstore.numpy_store import NumpyStore

        store = NumpyStore(int(vectors.shape[1]))
    t = time.perf_counter()
    store.upsert(records, vectors)
    out["upsert_s"] = round(time.perf_counter() - t, 3)
    out["index_mem_mib"] = mib(sampler.used_now() - mem_before_store)
    out["index_disk_mib"] = mib(_dir_bytes(db_dir)) if job["store"] == "chroma" else None
    out["index_nbytes_mib"] = mib(store.nbytes()) if job["store"] == "numpy" else None
    del vectors

    queries = job["queries"]
    qvecs = np.load(job["query_vectors"]) if job.get("query_vectors") else None
    filters = [{s: MemoryFilter.model_validate(f) for s, f in q["filters"].items()} for q in queries]
    scenarios = list(queries[0]["filters"]) if queries else []
    ks = [int(k) for k in job["ks"]]

    def embed(i: int):
        if qvecs is not None:
            return qvecs[i], None
        t = time.perf_counter()
        v = embedder.embed_queries([queries[i]["text"]])[0]
        sync()
        return v, (time.perf_counter() - t) * 1000

    def one(i: int, scenario: str, k: int, vec):
        t = time.perf_counter()
        hits = store.search(vec, k, filters[i][scenario])
        t_search = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        store.get([h.memory_id for h in hits])
        return hits, t_search, (time.perf_counter() - t) * 1000

    # cold: the very first query, including first-call overheads
    if queries:
        t = time.perf_counter()
        vec, _ = embed(0)
        one(0, scenarios[0], max(ks), vec)
        out["cold_query_ms"] = round((time.perf_counter() - t) * 1000, 2)
        for i in range(min(job.get("warmup", 5), len(queries))):
            vec, _ = embed(i)
            for s in scenarios:
                one(i, s, max(ks), vec)

    samples = {s: {str(k): {"embed": [], "search": [], "get": []} for k in ks} for s in scenarios}
    hits_out = {s: {str(k): {} for k in ks} for s in scenarios}
    for rep in range(int(job.get("repeats", 3))):
        for i, q in enumerate(queries):
            vec, t_embed = embed(i)
            for s in scenarios:
                for k in ks:
                    hits, t_search, t_get = one(i, s, k, vec)
                    cell = samples[s][str(k)]
                    cell["embed"].append(t_embed)
                    cell["search"].append(t_search)
                    cell["get"].append(t_get)
                    if rep == 0:
                        hits_out[s][str(k)][q["id"]] = [h.memory_id for h in hits]
    out["samples_ms"] = samples
    out["hits"] = hits_out
    if job["store"] == "numpy":
        out["matched"] = {s: {q["id"]: store.match_count(filters[i][s]) for i, q in enumerate(queries)}
                          for s in scenarios}

    out["sys_mem_peak_mib"] = mib(sampler.stop())
    out["process_peak_rss_mib"] = mib(peak_rss())
    out["cuda_peak_mib"] = mib(torch.cuda.max_memory_allocated()) if torch is not None else None
    if emb_cfg and not emb_cfg.get("fake"):
        import torch as torch_mod

        out["torch_threads"] = torch_mod.get_num_threads()
    if job["store"] == "chroma":
        store.close()
        shutil.rmtree(db_dir, ignore_errors=True)
    return out


def _embed_only(job: dict) -> dict:
    """Vectors for the scale test, computed here so the parent never loads the model."""
    import numpy as np

    embedder = _embedder(job["embedding"])
    np.save(job["out_documents"], np.asarray(embedder.embed_documents(job["documents"]), dtype=np.float32))
    np.save(job["out_queries"], np.asarray(embedder.embed_queries(job["queries"]), dtype=np.float32))
    return {"ok": True}


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result = _embed_only(job) if job.get("mode") == "embed" else _measure(job)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
