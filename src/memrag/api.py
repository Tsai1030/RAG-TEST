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


def load_dialogues():
    from memrag.domain.schemas.dialogue import DialogueLog
    from memrag.infrastructure.storage.jsonl_repo import read_jsonl

    return read_jsonl(bootstrap.project_root() / bootstrap.DIALOGUES, DialogueLog)


def load_memories():
    from memrag.domain.schemas.memory import MemoryRecord
    from memrag.infrastructure.storage.jsonl_repo import read_jsonl

    return read_jsonl(bootstrap.project_root() / bootstrap.MEMORIES_DIR / "memories.jsonl", MemoryRecord)


def validate_dialogues() -> dict:
    from memrag.application.evaluation.corpus_check import check_dialogues

    logs = load_dialogues()
    report = check_dialogues(load_gold_set(), logs)
    return {"ok": report.ok, "issues": [asdict(i) for i in report.issues], "logs": len(logs),
            "turns": sum(len(lg.turns) for lg in logs)}


def _user_handles(gs) -> dict[str, str]:
    from memrag.application.summarize import user_handle

    return {p.person_id: user_handle([f.statement for f in gs.facts if f.person_id == p.person_id],
                                     p.name.split()[0] + "さん") for p in gs.personas}


def _summary_dict(result) -> dict:
    return {
        "log_id": result.log_id, "ok": result.ok, "first_pass_ok": result.first_pass_ok,
        "attempts": [asdict(a) for a in result.attempts],
        "memories": [r.model_dump(mode="json") for r in result.records],
    }


def summarize_dialogue(log_id: str, fake: bool = False, use_cache: bool = True) -> dict:
    """Summarize one dialogue log into memories, with every attempt and its validation errors."""
    from memrag.application.summarize import summarize_log
    from memrag.domain.errors import ConfigError

    log = next((lg for lg in load_dialogues() if lg.log_id == log_id), None)
    if log is None:
        raise ConfigError(f"dialogue {log_id!r} not found in {bootstrap.DIALOGUES}")
    cfg = bootstrap.pipeline_config()
    cache = bootstrap.response_cache(cfg) if use_cache and not fake else None
    result = summarize_log(log, _user_handles(load_gold_set())[log.person_id], bootstrap.llm_provider(cfg, fake),
                           bootstrap.prompt_store(), cache, cfg["summarize"]["prompt_version"])
    return _summary_dict(result)


def build_memory_corpus(fake: bool = False, limit: int | None = None, note: str = "") -> dict:
    """Summarize every dialogue log (or the first `limit`) into the memory corpus.

    Each run writes attempts.jsonl, failures.jsonl, memories.jsonl and manifest.json to
    results/runs/<run_id>/. A full real run also writes data/02_memories/, the frozen
    corpus, with its sha256 in corpus_manifest.json.
    """
    import hashlib
    import json

    from memrag.application.evaluation.corpus_check import corpus_stats
    from memrag.application.summarize import summarize_log
    from memrag.infrastructure.storage.jsonl_repo import write_jsonl
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    root = bootstrap.project_root()
    cfg = bootstrap.pipeline_config(root)
    gs = load_gold_set()
    logs = load_dialogues()[:limit] if limit else load_dialogues()
    llm = bootstrap.llm_provider(cfg, fake)
    prompts = bootstrap.prompt_store(root)
    version = cfg["summarize"]["prompt_version"]
    cache = None if fake else bootstrap.response_cache(cfg, root)
    handles = _user_handles(gs)
    run_dir = new_run_dir(root, "summarize" + ("_fake" if fake else ""))
    manifest = {
        "run_id": run_dir.name, "kind": "summarize", "started": datetime.now().isoformat(timespec="seconds"),
        "note": note, "fake": fake, "logs": len(logs), "llm": {**cfg["llm"], **llm.info()},
        "served_model": None if fake else llm.served_model(),
        "prompts": {n: prompts.get(n, version).sha256 for n in ("summarize", "summarize_retry")},
        "dialogues_sha256": hashlib.sha256((root / bootstrap.DIALOGUES).read_bytes()).hexdigest(),
        "environment": environment_info(root),
    }
    write_json(run_dir / "manifest.json", manifest)
    results = []
    for log in logs:
        result = summarize_log(log, handles[log.person_id], llm, prompts, cache, version)
        results.append(result)
        row = _summary_dict(result)
        append_jsonl(run_dir / "attempts.jsonl", row)
        if not result.ok:
            append_jsonl(run_dir / "failures.jsonl", {"log_id": log.log_id,
                                                       "errors": result.attempts[-1].errors})
    records = [r for res in results for r in res.records]
    write_jsonl(run_dir / "memories.jsonl", records)
    stats = corpus_stats(logs, results)
    manifest.update(finished=datetime.now().isoformat(timespec="seconds"), stats=stats)
    write_json(run_dir / "manifest.json", manifest)
    frozen = None
    if not fake and limit is None:
        out = root / bootstrap.MEMORIES_DIR
        write_jsonl(out / "memories.jsonl", records)
        frozen = hashlib.sha256((out / "memories.jsonl").read_bytes()).hexdigest()
        write_json(out / "corpus_manifest.json", {
            "run_id": run_dir.name, "memories_sha256": frozen, "dialogues_sha256": manifest["dialogues_sha256"],
            "llm": manifest["llm"], "served_model": manifest["served_model"], "prompts": manifest["prompts"],
            "stats": {k: v for k, v in stats.items() if k != "uncovered_facts"},
            "uncovered_facts": stats["uncovered_facts"]})
    return {"run_dir": str(run_dir), "frozen_sha256": frozen, **stats}


def evaluate_retrieval(selection: str = "all", split: str | None = "dev", device: str = "auto",
                       fake: bool = False, note: str = "", corpus: str = "gold") -> dict:
    """Embed the corpus and queries with each model and score exact cosine retrieval.

    corpus="gold": the gold fact statements themselves (ideal summaries), so the result
    measures the embedding model alone. corpus="memories": the frozen memories summarized
    from the dialogue logs (data/02_memories/). Every model runs in float32 (plan 5.1.8).
    Writes manifest.json, metrics.json, per_query.jsonl and vectors/<model>.npz
    to results/runs/<run_id>/.
    """
    import time

    import numpy as np

    from memrag.application.evaluation.retrieval_eval import PRIMARY_SCOPE, rank
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    root = bootstrap.project_root()
    reg = bootstrap.registry(root)
    specs = reg.select(selection)
    data, corpus_info = _retrieval_input(corpus, split)
    run_dir = new_run_dir(root, f"retrieval_eval_{corpus}" + ("_fake" if fake else ""))
    (run_dir / "vectors").mkdir()
    device = "cpu" if fake else bootstrap.resolve_device(device)
    manifest = {
        "run_id": run_dir.name,
        "kind": "retrieval_eval",
        "started": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        **corpus_info,
        "queries": len(data.queries),
        "docs": len(data.docs),
        "queries_unanswerable_in_corpus": data.unanswerable_in_corpus,
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
        doc_vecs = embedder.embed_documents([d.text for d in data.docs])
        doc_s = time.perf_counter() - start
        query_vecs = embedder.embed_queries([q.text for q in data.queries])
        del embedder
        bootstrap.release_accelerator_memory()
        np.savez_compressed(run_dir / "vectors" / f"{spec.name}.npz", documents=doc_vecs, queries=query_vecs,
                            doc_ids=[d.doc_id for d in data.docs], query_ids=[q.query_id for q in data.queries])
        rows = rank(query_vecs, doc_vecs, data)
        for row in rows:
            append_jsonl(run_dir / "per_query.jsonl", {"model": spec.name, **row})
        per_model[spec.name] = rows
        timing[spec.name] = {"documents_per_s": round(len(data.docs) / doc_s, 1)}
    metrics = _metrics(per_model, timing)
    write_json(run_dir / "metrics.json", metrics)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "primary_scope": PRIMARY_SCOPE, **metrics}


def _retrieval_input(corpus: str, split: str | None):
    """The evaluation input and the manifest fields that identify the corpus and its labels."""
    import hashlib

    from memrag.application.evaluation.retrieval_eval import build_input
    from memrag.domain.errors import ConfigError
    from memrag.domain.schemas.evaluation import MemoryLabel, Split
    from memrag.infrastructure.storage.jsonl_repo import read_jsonl

    if corpus not in ("gold", "memories"):
        raise ConfigError(f"corpus must be 'gold' or 'memories', got {corpus!r}")
    split_enum = Split(split) if split else None
    info: dict = {"corpus_name": corpus, "split": split or "all"}
    if corpus == "gold":
        info["corpus"] = "gold_facts.statement"
        return build_input(load_gold_set(), split_enum), info
    folder = bootstrap.project_root() / bootstrap.MEMORIES_DIR
    pool_path = folder / "pool_labels.jsonl"
    pool = read_jsonl(pool_path, MemoryLabel) if pool_path.is_file() else []
    info.update(
        corpus="data/02_memories/memories.jsonl",
        corpus_sha256=hashlib.sha256((folder / "memories.jsonl").read_bytes()).hexdigest(),
        pool_labels=len(pool),
        pool_labels_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest() if pool else None,
    )
    return build_input(load_gold_set(), split_enum, load_memories(), pool), info


def _metrics(per_model: dict[str, list[dict]], timing: dict[str, dict] | None = None) -> dict:
    from memrag.application.evaluation.retrieval_eval import compare, summarize

    timing = timing or {}
    return {
        "models": {m: {**summarize(rows), **({"timing": timing[m]} if m in timing else {})}
                   for m, rows in per_model.items()},
        "compare_ndcg@10": compare(per_model),
    }


def rescore_retrieval(run_dir: str, note: str = "") -> dict:
    """Score a finished evaluation again with the current labels, reusing its saved vectors.

    No model is loaded, so it runs anywhere (also on Windows). Refuses when the corpus or
    the queries changed since the run. Writes a new results/runs/<run_id>_retrieval_rescore/.
    """
    import json

    import numpy as np

    from memrag.application.evaluation.retrieval_eval import PRIMARY_SCOPE, rank
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import append_jsonl, new_run_dir, write_json

    source = Path(run_dir)
    old = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    corpus = old.get("corpus_name") or ("memories" if old["corpus"].startswith("data/02") else "gold")
    data, corpus_info = _retrieval_input(corpus, None if old["split"] == "all" else old["split"])
    if old.get("corpus_sha256") != corpus_info.get("corpus_sha256"):
        raise ConfigError(f"{source.name}: the corpus changed since this run; run `memrag eval run` again")
    root = bootstrap.project_root()
    out = new_run_dir(root, "retrieval_rescore")
    per_model = {}
    order = list(old.get("models") or {})
    paths = sorted((source / "vectors").glob("*.npz"),
                   key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.stem))
    for path in paths:
        vec = np.load(path)
        if list(vec["doc_ids"]) != [d.doc_id for d in data.docs] or \
                list(vec["query_ids"]) != [q.query_id for q in data.queries]:
            raise ConfigError(f"{path.name}: documents or queries differ from the current data")
        rows = rank(vec["queries"], vec["documents"], data)
        for row in rows:
            append_jsonl(out / "per_query.jsonl", {"model": path.stem, **row})
        per_model[path.stem] = rows
    metrics = _metrics(per_model)
    write_json(out / "metrics.json", metrics)
    write_json(out / "manifest.json", {
        "run_id": out.name, "kind": "retrieval_rescore", "source_run": source.name, "note": note,
        "created": datetime.now().isoformat(timespec="seconds"), **corpus_info,
        "queries": len(data.queries), "docs": len(data.docs),
        "queries_unanswerable_in_corpus": data.unanswerable_in_corpus, "search": old.get("search"),
        "primary_scope": PRIMARY_SCOPE, "models": old.get("models")})
    return {"run_dir": str(out), "primary_scope": PRIMARY_SCOPE, **metrics}


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
