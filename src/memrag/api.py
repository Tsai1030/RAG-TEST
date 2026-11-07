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


def load_gold_set(include_second: bool = False):
    """Personas, gold facts, queries and first-pass labels from data/00_gold/."""
    from memrag.application.evaluation.dataset_check import GoldSet
    from memrag.domain.schemas.evaluation import GoldFact, Persona, QueryLabel, TestQuery
    from memrag.infrastructure.storage.jsonl_repo import read_jsonl

    labels = read_jsonl(bootstrap.gold_path("labels"), QueryLabel)
    if include_second:
        labels += read_jsonl(bootstrap.gold_path("labels_second"), QueryLabel)
    return GoldSet(
        personas=read_jsonl(bootstrap.gold_path("personas"), Persona),
        facts=read_jsonl(bootstrap.gold_path("facts"), GoldFact),
        queries=read_jsonl(bootstrap.gold_path("queries"), TestQuery),
        labels=labels,
    )


def validate_dataset() -> dict:
    from memrag.application.evaluation.dataset_check import check_gold_set, stats

    gs = load_gold_set()
    report = check_gold_set(gs)
    return {"ok": report.ok, "issues": [asdict(i) for i in report.issues], "stats": stats(gs)}


def show_query(query_id: str) -> dict:
    """A query with its filters and every fact labelled relevant, for manual checking."""
    gs = load_gold_set()
    query = next((q for q in gs.queries if q.query_id == query_id), None)
    if query is None:
        from memrag.domain.errors import ConfigError

        raise ConfigError(f"query {query_id!r} not found in {bootstrap.gold_path('queries')}")
    facts = {f.fact_id: f for f in gs.facts}
    persona = next(p for p in gs.personas if p.person_id == query.person_id)
    relevant = sorted((l for l in gs.labels if l.query_id == query_id), key=lambda l: -l.grade)
    return {
        "query": query.model_dump(mode="json"),
        "persona": persona.model_dump(mode="json"),
        "relevant": [{"grade": l.grade, **facts[l.fact_id].model_dump(mode="json")} for l in relevant],
    }


def assign_dataset_splits(write: bool = False) -> dict:
    """Stratified dev/test split by query type; writes queries.jsonl only when `write` is true."""
    from memrag.application.evaluation.dataset_check import assign_splits
    from memrag.infrastructure.storage.jsonl_repo import write_jsonl

    gs = load_gold_set()
    splits = assign_splits(gs.queries)
    changed = [q.query_id for q in gs.queries if q.split != splits[q.query_id]]
    if write and changed:
        write_jsonl(bootstrap.gold_path("queries"),
                    [q.model_copy(update={"split": splits[q.query_id]}) for q in gs.queries])
    counts: dict[str, int] = {}
    for s in splits.values():
        counts[s.value] = counts.get(s.value, 0) + 1
    return {"counts": counts, "changed": len(changed), "written": bool(write and changed)}


def make_second_pass_sample(write: bool = False) -> list[str]:
    """Stratified 25% of queries for the independent second annotation (D7)."""
    import json

    from memrag.application.evaluation.dataset_check import second_pass_sample

    sample = second_pass_sample(load_gold_set().queries)
    if write:
        bootstrap.gold_path("second_sample").write_text(json.dumps(sample, indent=1) + "\n", encoding="utf-8")
    return sample


def annotation_agreement(first: str = "claude-1", second: str = "claude-2") -> dict:
    """Cohen's kappa between the two annotation passes on the second-pass sample."""
    import json

    from memrag.application.evaluation.dataset_check import agreement

    sample = json.loads(bootstrap.gold_path("second_sample").read_text(encoding="utf-8"))
    return agreement(load_gold_set(include_second=True), first, second, sample)


def evaluate_retrieval(selection: str = "all", split: str | None = "dev", device: str = "auto",
                       fake: bool = False, note: str = "") -> dict:
    """Embed the gold facts and queries with each model and score exact cosine retrieval.

    The corpus is the gold fact statements themselves (ideal summaries), so the result
    measures the embedding model alone. Every model runs in float32 (plan 5.1.8).
    Writes manifest.json, metrics.json, per_query.jsonl and vectors/<model>.npz
    to results/runs/<run_id>/.
    """
    import time

    import numpy as np

    from memrag.application.evaluation.retrieval_eval import PRIMARY_SCOPE, build_input, compare, rank, summarize
    from memrag.domain.schemas.evaluation import Split
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    root = bootstrap.project_root()
    reg = bootstrap.registry(root)
    specs = reg.select(selection)
    data = build_input(load_gold_set(), Split(split) if split else None)
    run_dir = new_run_dir(root, "retrieval_eval" + ("_fake" if fake else ""))
    (run_dir / "vectors").mkdir()
    device = "cpu" if fake else bootstrap.resolve_device(device)
    manifest = {
        "run_id": run_dir.name,
        "kind": "retrieval_eval",
        "started": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "corpus": "gold_facts.statement",
        "split": split or "all",
        "queries": len(data.queries),
        "facts": len(data.facts),
        "search": "exact cosine",
        "primary_scope": PRIMARY_SCOPE,
        "device": device,
        "fake": fake,
        "models": {s.name: {"hf_id": s.hf_id, "revision": s.revision, "dtype": "float32",
                            "query_prefix": s.query_prefix, "document_prefix": s.document_prefix} for s in specs},
        "environment": environment_info(root),
    }
    write_json(run_dir / "manifest.json", manifest)
    per_model: dict[str, list[dict]] = {}
    timing: dict[str, dict] = {}
    for spec in specs:
        embedder = bootstrap.embedding_provider(reg, spec.name, device=device, fake=fake, dtype="float32")
        start = time.perf_counter()
        doc_vecs = embedder.embed_documents([f.statement for f in data.facts])
        doc_s = time.perf_counter() - start
        query_vecs = embedder.embed_queries([q.text for q in data.queries])
        del embedder
        bootstrap.release_accelerator_memory()
        np.savez_compressed(run_dir / "vectors" / f"{spec.name}.npz", documents=doc_vecs, queries=query_vecs,
                            fact_ids=[f.fact_id for f in data.facts], query_ids=[q.query_id for q in data.queries])
        rows = rank(query_vecs, doc_vecs, data)
        for row in rows:
            append_jsonl(run_dir / "per_query.jsonl", {"model": spec.name, **row})
        per_model[spec.name] = rows
        timing[spec.name] = {"documents_per_s": round(len(data.facts) / doc_s, 1)}
    metrics = {
        "models": {m: {**summarize(rows), "timing": timing[m]} for m, rows in per_model.items()},
        "compare_ndcg@10": compare(per_model),
    }
    write_json(run_dir / "metrics.json", metrics)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "primary_scope": PRIMARY_SCOPE, **metrics}


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
