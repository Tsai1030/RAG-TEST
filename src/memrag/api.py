"""Facade: the stable public API. CLI, UI and scripts only call this module."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from memrag import bootstrap
from memrag.domain.ports import EmbeddingProvider
from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind

if TYPE_CHECKING:
    from datetime import date

    from memrag.application.answer import Answer
    from memrag.application.ingest import IngestReport
    from memrag.domain.schemas.dialogue import DialogueLog
    from memrag.domain.schemas.memory import MemoryRecord
    from memrag.domain.schemas.retrieval import MemoryFilter


class MemRAG:
    """The pipeline for one embedding model: ingest logs, search, ask (plan 6.1).

    The vector store and the LLM are opened on first use, so embedding alone needs neither.
    With `fake`, the embedder and the LLM are fakes and the store lives in memory.
    """

    def __init__(self, embedder: EmbeddingProvider, config: dict, fake: bool = False, store_dir: Path | None = None,
                 use_cache: bool = True):
        self._embedder = embedder
        self._cfg = config
        self._fake = fake
        self._store_dir = store_dir
        self._use_cache = use_cache and not fake
        self._store = self._llm = None

    @classmethod
    def from_config(cls, embedding_model: str, device: str = "auto", fake: bool = False,
                    store_dir: str | Path | None = None, use_cache: bool = True) -> MemRAG:
        reg = bootstrap.registry()
        return cls(bootstrap.embedding_provider(reg, embedding_model, device=device, fake=fake),
                   bootstrap.pipeline_config(), fake, Path(store_dir) if store_dir else None, use_cache)

    @property
    def embedding_model(self) -> ModelSpec:
        return self._embedder.info()

    def embed(self, texts: Sequence[str], kind: TextKind = "query"):
        if kind == "query":
            return self._embedder.embed_queries(texts)
        return self._embedder.embed_documents(texts)

    def explain_embedding(self, text: str, kind: TextKind = "query") -> EncodedText:
        return self._embedder.explain(text, kind)

    @property
    def store(self):
        if self._store is None:
            spec = self.embedding_model
            self._store = bootstrap.vector_store(spec.name, spec.dim, fake=self._fake, path=self._store_dir)
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = bootstrap.llm_provider(self._cfg, self._fake)
        return self._llm

    @property
    def threshold(self) -> float | None:
        """configs/thresholds.yaml for this model; None when uncalibrated or switched off."""
        if not self._cfg["answer"]["use_threshold"]:
            return None
        return bootstrap.thresholds().get("thresholds", {}).get(self.embedding_model.name)

    def _cache(self):
        return bootstrap.response_cache(self._cfg) if self._use_cache else None

    def index(self, memories: Sequence[MemoryRecord]) -> int:
        """Embed and upsert memories (idempotent); returns the store size."""
        if memories:
            self.store.upsert(memories, self._embedder.embed_documents([m.content for m in memories]))
        return self.store.count()

    def ingest_log(self, log: DialogueLog | str | Path, user_name: str | None = None) -> IngestReport:
        """Summarize one dialogue log, embed the memories and write them to the store.

        `log` is a DialogueLog or the path of one as JSON. Nothing is written when the
        summary fails validation after the retries; the report says why.
        """
        from memrag.application.ingest import ingest_log
        from memrag.domain.errors import LLMUnavailableError, PipelineError
        from memrag.domain.schemas.dialogue import DialogueLog

        if not isinstance(log, DialogueLog):
            log = DialogueLog.model_validate_json(Path(log).read_text(encoding="utf-8"))
        if user_name is None:
            user_name = _user_handles(load_gold_set()).get(log.person_id, "ユーザーさん")
        try:
            return ingest_log(log, user_name, self._embedder, self.store, self.llm, bootstrap.prompt_store(),
                              self._cache(), self._cfg["summarize"]["prompt_version"])
        except LLMUnavailableError as e:
            raise PipelineError("summarize", log.log_id, str(e)) from e

    def search(self, text: str, filters: MemoryFilter | None = None, top_k: int = 5) -> list[tuple[MemoryRecord, float]]:
        """Filtered top-k as (memory, cosine score), best first. The filter is used as given."""
        hits = self.store.search(self._embedder.embed_queries([text])[0], top_k, filters)
        records = {r.memory_id: r for r in self.store.get([h.memory_id for h in hits])}
        return [(records[h.memory_id], h.score) for h in hits]

    def pipeline_filter(self, filters: MemoryFilter | None) -> MemoryFilter:
        """The filter `ask` uses: the 5.1.6 decisions in pipeline.yaml (date fallback, no category)."""
        from memrag.domain.schemas.retrieval import MemoryFilter

        r = self._cfg["retrieval"]
        flt = filters or MemoryFilter()
        return flt.model_copy(update={"date_fallback": r["date_fallback"],
                                      **({} if r["category_filter"] else {"category": None})})

    def ask(self, question: str, filters: MemoryFilter | None = None, today: date | None = None,
            use_threshold: bool = True, threshold_mode: str | None = None) -> Answer:
        """Answer from the memories with cited memory_ids, the trace and per-stage timings.

        `threshold_mode` (per_memory | gate) defaults to pipeline.yaml's answer.threshold_mode.
        """
        from datetime import date as date_cls

        from memrag.application.answer import AskSettings, ask
        from memrag.domain.errors import LLMUnavailableError, PipelineError

        a = self._cfg["answer"]
        settings = AskSettings(self._cfg["retrieval"]["top_k"], self.threshold if use_threshold else None,
                               a["context_budget_chars"], self._cfg["refusal_text"], a["prompt_version"],
                               threshold_mode or a["threshold_mode"])
        try:
            return ask(question, self.pipeline_filter(filters), today or date_cls.today(), self._embedder,
                       self.store, self.llm, bootstrap.prompt_store(), settings, self._cache())
        except LLMUnavailableError as e:
            raise PipelineError("answer", repr(question), str(e)) from e

    def manifest(self) -> dict:
        """What identifies a pipeline run: model, threshold, prompts, settings, LLM."""
        spec = self.embedding_model
        prompts = bootstrap.prompt_store()
        th = bootstrap.thresholds()
        return {
            "embedding_model": {"name": spec.name, "hf_id": spec.hf_id, "revision": spec.revision, "dim": spec.dim,
                                "query_prefix": spec.query_prefix, "document_prefix": spec.document_prefix},
            "threshold": {"value": self.threshold, "calibration_run": th.get("calibration_run"),
                          "uncalibrated": self.threshold is None},
            "prompts": {n: {"version": v, "sha256": prompts.get(n, v).sha256} for n, v in (
                ("summarize", self._cfg["summarize"]["prompt_version"]),
                ("answer", self._cfg["answer"]["prompt_version"]))},
            "retrieval": self._cfg["retrieval"], "answer": self._cfg["answer"],
            "llm": {**self._cfg["llm"], **self.llm.info()}, "fake": self._fake,
        }

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


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


def summarize_dialogue(log_id: str | None = None, fake: bool = False, use_cache: bool = True,
                       log_json: str | None = None) -> dict:
    """Summarize one dialogue log into memories, with every attempt and its validation errors.

    The log is one of data/01_dialogues by id, or `log_json` (a pasted log).
    """
    from pydantic import ValidationError

    from memrag.application.summarize import summarize_log
    from memrag.domain.errors import ConfigError
    from memrag.domain.schemas.dialogue import DialogueLog

    if log_json:
        try:
            log = DialogueLog.model_validate_json(log_json)
        except ValidationError as e:
            raise ConfigError(f"the pasted log is not a valid dialogue: {e}") from e
    else:
        log = next((lg for lg in load_dialogues() if lg.log_id == log_id), None)
        if log is None:
            raise ConfigError(f"dialogue {log_id!r} not found in {bootstrap.DIALOGUES}")
    cfg = bootstrap.pipeline_config()
    cache = bootstrap.response_cache(cfg) if use_cache and not fake else None
    result = summarize_log(log, _user_handles(load_gold_set()).get(log.person_id, "ユーザーさん"),
                           bootstrap.llm_provider(cfg, fake), bootstrap.prompt_store(), cache,
                           cfg["summarize"]["prompt_version"])
    return _summary_dict(result)


FEEDBACK = Path("results") / "feedback" / "query_candidates.jsonl"


def save_feedback(text: str, person_id: str, problem: str, model: str = "", expected: str = "") -> dict:
    """Store a problem query found by hand as a candidate. Candidates join the query set only after review."""
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import append_jsonl

    if not text.strip() or not person_id.strip():
        raise ConfigError("a feedback candidate needs the query text and the person_id")
    path = bootstrap.project_root() / FEEDBACK
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"created": datetime.now().isoformat(timespec="seconds"), "text": text.strip(), "person_id": person_id.strip(),
           "problem": problem.strip(), "model": model, "expected": expected.strip(), "status": "unreviewed"}
    append_jsonl(path, row)
    return {"path": str(path), **row}


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


def _load_run(run_dir: str):
    """A finished `eval run`: its manifest, the current evaluation input, and each model's saved vectors.

    Refuses when the corpus, documents or queries changed since the run.
    """
    import json

    import numpy as np

    from memrag.domain.errors import ConfigError

    source = Path(run_dir)
    old = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    corpus = old.get("corpus_name") or ("memories" if old["corpus"].startswith("data/02") else "gold")
    data, corpus_info = _retrieval_input(corpus, None if old["split"] == "all" else old["split"])
    if old.get("corpus_sha256") != corpus_info.get("corpus_sha256"):
        raise ConfigError(f"{source.name}: the corpus changed since this run; run `memrag eval run` again")
    order = list(old.get("models") or {})
    paths = sorted((source / "vectors").glob("*.npz"),
                   key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.stem))
    vectors = {}
    for path in paths:
        vec = np.load(path)
        if list(vec["doc_ids"]) != [d.doc_id for d in data.docs] or \
                list(vec["query_ids"]) != [q.query_id for q in data.queries]:
            raise ConfigError(f"{path.name}: documents or queries differ from the current data")
        vectors[path.stem] = (vec["queries"], vec["documents"])
    return source, old, data, corpus_info, vectors


def calibrate_thresholds(run_dir: str, write: bool = False, note: str = "") -> dict:
    """Calibrate each model's score threshold on small candidate sets of a dev-split memory run.

    Writes results/runs/<run_id>_threshold_calibration/; with `write`, also configs/thresholds.yaml,
    which must be committed before the test set is evaluated (plan 5.1.8).
    """
    import yaml

    from memrag.application.evaluation import thresholds as th
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import new_run_dir, write_json

    source, old, data, corpus_info, vectors = _load_run(run_dir)
    if corpus_info["split"] != "dev" or corpus_info["corpus_name"] != "memories":
        raise ConfigError("calibrate on a dev-split run with --corpus memories")
    sets = th.candidate_sets(data)
    results = {}
    for model, (query_vecs, doc_vecs) in vectors.items():
        t = th.calibrate(query_vecs, doc_vecs, sets)
        results[model] = th.evaluate(query_vecs, doc_vecs, data, sets, t)
    settings = {"target_false_recall": th.TARGET_FALSE_RECALL, "sizes": list(th.SIZES),
                "samples_per_size": th.SAMPLES_PER_SIZE, "seed": th.SEED}
    root = bootstrap.project_root()
    out = new_run_dir(root, "threshold_calibration")
    write_json(out / "metrics.json", {"models": results})
    write_json(out / "manifest.json", {
        "run_id": out.name, "kind": "threshold_calibration", "source_run": source.name, "note": note,
        "created": datetime.now().isoformat(timespec="seconds"), **corpus_info, **settings,
        "positive_sets": sum(s.positive for s in sets), "negative_sets": sum(not s.positive for s in sets)})
    if write:
        config = {
            "calibrated_on": source.name, "calibration_run": out.name, "corpus_sha256": corpus_info["corpus_sha256"],
            "pool_labels_sha256": corpus_info.get("pool_labels_sha256"), **settings,
            "thresholds": {m: r["threshold"] for m, r in results.items()},
        }
        header = "# Score thresholds per model (plan 5.1.8). Written by `memrag eval calibrate --write`;\n" \
                 "# commit before evaluating the test set and do not edit by hand.\n"
        (root / bootstrap.THRESHOLDS_CONFIG).write_text(
            header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"run_dir": str(out), "written": write, **settings, "models": results}


def evaluate_thresholds(run_dir: str, note: str = "") -> dict:
    """Small-candidate metrics of a memory run with the fixed thresholds in configs/thresholds.yaml."""
    import yaml

    from memrag.application.evaluation import thresholds as th
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import new_run_dir, write_json

    root = bootstrap.project_root()
    path = root / bootstrap.THRESHOLDS_CONFIG
    if not path.is_file():
        raise ConfigError(f"{path} not found; run `memrag eval calibrate <dev run> --write` first")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, old, data, corpus_info, vectors = _load_run(run_dir)
    if corpus_info.get("corpus_sha256") != config["corpus_sha256"]:
        raise ConfigError("the thresholds were calibrated on another corpus")
    sets = th.candidate_sets(data, config["sizes"], config["samples_per_size"], config["seed"])
    results = {}
    for model, (query_vecs, doc_vecs) in vectors.items():
        if model not in config["thresholds"]:
            raise ConfigError(f"no threshold for {model} in {path.name}")
        results[model] = th.evaluate(query_vecs, doc_vecs, data, sets, config["thresholds"][model])
    out = new_run_dir(root, "threshold_eval")
    write_json(out / "metrics.json", {"models": results})
    write_json(out / "manifest.json", {
        "run_id": out.name, "kind": "threshold_eval", "source_run": source.name, "note": note,
        "created": datetime.now().isoformat(timespec="seconds"), **corpus_info,
        "thresholds_from": config["calibration_run"]})
    return {"run_dir": str(out), "split": corpus_info["split"], "models": results}


def select_models(retrieval_run: str, threshold_run: str, probe_run: str, note: str = "") -> dict:
    """Plan 5.1.9: hard gates and weighted ranking from configs/selection.yaml.

    retrieval_run: test-split `eval run` or `eval rescore` on the memory corpus;
    threshold_run: `eval thresholds` on the same vectors; probe_run: `models probe`.
    """
    import json

    import yaml

    from memrag.application.evaluation import selection as sel
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import new_run_dir, write_json

    root = bootstrap.project_root()
    rules = yaml.safe_load((root / bootstrap.SELECTION_CONFIG).read_text(encoding="utf-8"))
    ev = rules["evaluation"]

    def load(path: str, name: str) -> dict:
        return json.loads((Path(path) / name).read_text(encoding="utf-8"))

    r_man, r_met = load(retrieval_run, "manifest.json"), load(retrieval_run, "metrics.json")
    t_man, t_met = load(threshold_run, "manifest.json"), load(threshold_run, "metrics.json")
    source = r_man.get("source_run", r_man["run_id"])
    if r_man["split"] != ev["split"] or r_man.get("corpus_name", "memories") != ev["corpus"]:
        raise ConfigError(f"{Path(retrieval_run).name}: selection needs split={ev['split']}, corpus={ev['corpus']}")
    if t_man["source_run"] != source:
        raise ConfigError("the threshold run must use the same vectors as the retrieval run")
    probe = {}
    for line in (Path(probe_run) / "load_probe.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("device") == "cuda" and row.get("ok"):
            probe[row["model"]] = row
    owners = {m.memory_id: m.person_id for m in load_memories()}
    askers = {q.query_id: q.person_id for q in load_gold_set().queries}
    cross: dict[str, int] = {}
    for line in (Path(retrieval_run) / "per_query.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scope"] == ev["scope"]:
            cross[row["model"]] = cross.get(row["model"], 0) + sum(
                owners[t[0]] != askers[row["query_id"]] for t in row["top"])
    reg = bootstrap.registry(root)
    facts = []
    for name, m in r_met["models"].items():
        spec = reg.get(name)
        s, th = m[ev["scope"]], t_met["models"][name]["by_size"]["all"]
        facts.append(sel.ModelFacts(
            name=name, role=spec.role, commercial_ok=spec.commercial_ok,
            metrics={"ndcg@10": s["ndcg@10"]["mean"], "recall@5": s["recall@5"]["mean"],
                     "small_set_false_block": th["false_block"]["rate"],
                     "query_latency_p95_ms": probe[name]["latency_ms_p95"],
                     "memory_peak_mib": probe[name]["sys_mem_peak_mib"]},
            cross_user_errors=cross.get(name, 0), false_recall_ci_low=th["false_recall"]["ci95"][0]))
    sel.apply_gates(facts, rules["hard_gates"])
    ranked = [f for f in facts if not f.gate_failures and (f.role == "candidate" or not rules["candidates_only"])]
    scores = sel.weighted_scores(ranked, rules["weights"])
    order = sorted(scores, key=lambda n: -scores[n]["score"])
    significance = r_met["compare_ndcg@10"][ev["scope"]]
    out = {
        "rules_version": rules["version"], "retrieval_run": Path(retrieval_run).name,
        "threshold_run": Path(threshold_run).name, "probe_run": Path(probe_run).name,
        "models": [{"name": f.name, "role": f.role, **f.metrics, "cross_user_errors": f.cross_user_errors,
                    "false_recall_ci_low": f.false_recall_ci_low, "gate_failures": f.gate_failures,
                    **({"score": scores[f.name]["score"], "normalised": scores[f.name]["normalised"],
                        "rank": order.index(f.name) + 1} if f.name in scores else {})} for f in facts],
        "ranking": order,
        "ndcg_best": significance["best"],
        "ndcg_vs_best": significance["vs_best"],
        "sensitivity": sel.sensitivity(ranked, rules["weights"]),
        "tie_rule": rules["tie_rule"],
    }
    run_dir = new_run_dir(root, "selection")
    write_json(run_dir / "selection.json", out)
    write_json(run_dir / "manifest.json", {"run_id": run_dir.name, "kind": "selection", "note": note,
                                           "created": datetime.now().isoformat(timespec="seconds"),
                                           "rules": str(bootstrap.SELECTION_CONFIG)})
    return {"run_dir": str(run_dir), **out}


def build_index(model: str, device: str = "auto", fake: bool = False) -> dict:
    """Embed the frozen memory corpus with `model` and upsert it into results/vector_db/ (idempotent)."""
    import time

    memories = load_memories()
    rag = MemRAG.from_config(model, device=device, fake=fake)
    start = time.perf_counter()
    vectors = rag.embed([m.content for m in memories], kind="document")
    embed_s = time.perf_counter() - start
    start = time.perf_counter()
    rag.store.upsert(memories, vectors)
    out = {"model": model, "memories": rag.store.count(), "embed_s": round(embed_s, 2),
           "upsert_s": round(time.perf_counter() - start, 2), "path": "（記憶體）" if fake else str(bootstrap.VECTOR_DB)}
    rag.close()
    return out


_HIT_FIELDS = {"memory_id", "person_id", "category", "event_date", "conversation_date", "content"}


def _open_index(model: str, device: str, fake: bool) -> MemRAG:
    """The model's index; a fake run indexes the frozen corpus in memory first."""
    from memrag.domain.errors import ConfigError

    rag = MemRAG.from_config(model, device=device, fake=fake)
    if fake:
        rag.index(load_memories())
    elif rag.store.count() == 0:
        rag.close()
        raise ConfigError(f"no index for {model}; run `memrag store build --model {model}` first")
    return rag


def search_memories(text: str, model: str, person_id: str | None = None, category: str | None = None,
                    date_from: str | None = None, date_to: str | None = None, k: int = 5,
                    device: str = "auto", fake: bool = False) -> dict:
    """Search the index built by `build_index`; the query gets the model's query prefix."""
    from memrag.domain.schemas.retrieval import MemoryFilter

    flt = MemoryFilter(person_id=person_id, category=category, date_from=date_from, date_to=date_to)
    rag = _open_index(model, device, fake)
    hits = rag.search(text, flt, k)
    rag.close()
    return {
        "query": rag.explain_embedding(text, "query").encoded_text, "model": model,
        "filter": flt.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        "hits": [{"score": round(s, 4), **m.model_dump(mode="json", include=_HIT_FIELDS)} for m, s in hits],
    }


def answer_dict(answer: Answer, records: dict | None = None) -> dict:
    """An Answer as JSON-ready data; `records` adds the memory text to each retrieved hit."""
    out = asdict(answer)
    out["filter"] = answer.filter.model_dump(mode="json", exclude_defaults=True)
    out["uncalibrated"] = answer.uncalibrated
    if records:
        for hit in out["retrieved"]:
            hit.update(records[hit["memory_id"]].model_dump(mode="json", include=_HIT_FIELDS - {"memory_id"}))
    return out


def ask_question(text: str, model: str, person_id: str | None = None, date_from: str | None = None,
                 date_to: str | None = None, today: str | None = None, use_threshold: bool = True,
                 device: str = "auto", fake: bool = False, threshold_mode: str | None = None) -> dict:
    """One question through the whole pipeline, with the trace (plan 5.1.6, `memrag ask`)."""
    from datetime import date

    from memrag.domain.schemas.retrieval import MemoryFilter

    rag = _open_index(model, device, fake)
    answer = rag.ask(text, MemoryFilter(person_id=person_id, date_from=date_from, date_to=date_to),
                     date.fromisoformat(today) if today else None, use_threshold, threshold_mode)
    records = {r.memory_id: r for r in rag.store.get([h["memory_id"] for h in answer.retrieved])}
    out = {**answer_dict(answer, records), "manifest": rag.manifest()}
    rag.close()
    return out


def ingest_dialogues(model: str, log_ids: Sequence[str] = (), path: str | None = None, device: str = "auto",
                     fake: bool = False, note: str = "") -> dict:
    """Summarize dialogue logs and write their memories to the model's store (plan 5.1.6).

    `path`: one log as JSON; otherwise the logs in data/01_dialogues with these ids (all
    when empty). Writes reports.jsonl and manifest.json to results/runs/<run_id>_ingest/.
    """
    from memrag.domain.errors import ConfigError
    from memrag.domain.schemas.dialogue import DialogueLog
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    if path:
        logs = [DialogueLog.model_validate_json(Path(path).read_text(encoding="utf-8"))]
    else:
        logs = [lg for lg in load_dialogues() if not log_ids or lg.log_id in log_ids]
        missing = set(log_ids) - {lg.log_id for lg in logs}
        if missing:
            raise ConfigError(f"dialogues not found: {sorted(missing)}")
    root = bootstrap.project_root()
    rag = MemRAG.from_config(model, device=device, fake=fake)
    run_dir = new_run_dir(root, "ingest" + ("_fake" if fake else ""))
    manifest = {"run_id": run_dir.name, "kind": "ingest", "started": datetime.now().isoformat(timespec="seconds"),
                "note": note, "logs": len(logs), "source": path or str(bootstrap.DIALOGUES), **rag.manifest(),
                "environment": environment_info(root)}
    write_json(run_dir / "manifest.json", manifest)
    handles = _user_handles(load_gold_set())
    reports = []
    for log in logs:
        report = rag.ingest_log(log, handles.get(log.person_id, "ユーザーさん"))
        append_jsonl(run_dir / "reports.jsonl", asdict(report))
        reports.append(report)
    out = {"accepted": sum(r.accepted for r in reports), "failed": [r.log_id for r in reports if not r.ok],
           "retries": sum(r.retries for r in reports), "store_count": rag.store.count()}
    manifest.update(finished=datetime.now().isoformat(timespec="seconds"), **out)
    write_json(run_dir / "manifest.json", manifest)
    rag.close()
    return {"run_dir": str(run_dir), "logs": len(logs), **out,
            "reports": [asdict(r) for r in reports]}


def _update_pairs(gs, memories) -> dict[str, tuple[set[str], set[str]]]:
    """Queries about a changed fact: memories of the new fact and of the one it replaced."""
    replaced = {f.fact_id: f.supersedes for f in gs.facts if f.supersedes}
    grades: dict[str, dict[str, int]] = {}
    for label in gs.labels:
        grades.setdefault(label.query_id, {})[label.fact_id] = label.grade
    out = {}
    for qid, g in grades.items():
        for new, old in replaced.items():
            if g.get(new) == 2 and g.get(old):
                out[qid] = ({m.memory_id for m in memories if new in m.gold_fact_ids},
                            {m.memory_id for m in memories if old in m.gold_fact_ids})
    return out


# variant -> (use_threshold, threshold_mode); gate+llm reuses llm_only's cached LLM answers
# whenever the top memory passes, since the prompt is the same.
ANSWER_VARIANTS = {
    "threshold+llm": (True, "per_memory"),
    "gate+llm": (True, "gate"),
    "llm_only": (False, "per_memory"),
}


def evaluate_answers(model: str, split: str | None = None, limit: int | None = None, device: str = "auto",
                     fake: bool = False, note: str = "") -> dict:
    """Plan 5.1.6 acceptance on the query set: every query through `ask`, as of the day it was asked.

    Two variants: the plan's threshold + LLM, and the LLM alone (no threshold). The filter
    is the query's own, through `pipeline_filter`. Indexes the frozen corpus first
    (idempotent), checks that a second retrieval pass gives the same results, and writes
    answers.jsonl, metrics.json and manifest.json to results/runs/<run_id>_answer_eval/.
    """
    import hashlib

    from memrag.application.evaluation.answer_eval import score_answer, summarize
    from memrag.domain.schemas.retrieval import MemoryFilter
    from memrag.infrastructure.storage.manifest import append_jsonl, environment_info, new_run_dir, write_json

    root = bootstrap.project_root()
    gs = load_gold_set()
    memories = load_memories()
    data, corpus_info = _retrieval_input("memories", split)
    queries = data.queries[:limit] if limit else data.queries
    updates = _update_pairs(gs, memories)
    rag = MemRAG.from_config(model, device=device, fake=fake)
    rag.index(memories)
    records = {m.memory_id: m for m in memories}
    run_dir = new_run_dir(root, "answer_eval" + ("_fake" if fake else ""))
    manifest = {"run_id": run_dir.name, "kind": "answer_eval", "started": datetime.now().isoformat(timespec="seconds"),
                "note": note, **corpus_info, "queries": len(queries), "variants": list(ANSWER_VARIANTS),
                "store": "in-memory" if fake else str(bootstrap.VECTOR_DB), **rag.manifest(),
                "served_model": None if fake else rag.llm.served_model(),
                "dialogues_sha256": hashlib.sha256((root / bootstrap.DIALOGUES).read_bytes()).hexdigest(),
                "environment": environment_info(root)}
    write_json(run_dir / "manifest.json", manifest)
    rows: dict[str, list[dict]] = {v: [] for v in ANSWER_VARIANTS}
    first: dict[str, list] = {}
    for q in queries:
        f = q.filters
        flt = MemoryFilter(person_id=q.person_id, category=f.category, date_from=f.date_from, date_to=f.date_to)
        for variant, (use_threshold, mode) in ANSWER_VARIANTS.items():
            answer = rag.ask(q.text, flt, q.asked_on, use_threshold, mode)
            row = {"variant": variant, **score_answer(q, answer, data.grades[q.query_id], records,
                                                      updates.get(q.query_id)),
                   "timings_ms": answer.timings_ms, "cached": answer.cached}
            append_jsonl(run_dir / "answers.jsonl", row)
            rows[variant].append(row)
        first[q.query_id] = [(h["memory_id"], h["score"]) for h in answer.retrieved]
    repeat = [q.query_id for q in queries if first[q.query_id] != [
        (m.memory_id, round(s, 4)) for m, s in rag.search(q.text, rag.pipeline_filter(MemoryFilter(
            person_id=q.person_id, category=q.filters.category, date_from=q.filters.date_from,
            date_to=q.filters.date_to)), rag._cfg["retrieval"]["top_k"])]]
    metrics = {"variants": {v: summarize(r) for v, r in rows.items()},
               "by_split": {v: {s: summarize([r for r in rs if r["split"] == s]) for s in ("dev", "test")}
                            for v, rs in rows.items()},
               "retrieval_repeat": {"queries": len(queries), "different": repeat}}
    write_json(run_dir / "metrics.json", metrics)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    rag.close()
    return {"run_dir": str(run_dir), **metrics}


def check_vector_store(run_dir: str, note: str = "") -> dict:
    """Plan 5.1.5: ChromaDB (HNSW) against the exact store, reusing the vectors of a memory-corpus run.

    Per model: index build time and disk size, filter translation on query and random
    filters, cross-user hits, and HNSW recall@10. Writes results/runs/<run_id>_store_check/.
    """
    import shutil
    import tempfile
    import time

    from memrag.application.evaluation.store_check import compare
    from memrag.domain.errors import ConfigError
    from memrag.infrastructure.storage.manifest import new_run_dir, write_json
    from memrag.infrastructure.vectorstore.chroma_store import ChromaStore
    from memrag.infrastructure.vectorstore.exact_store import ExactStore

    source, old, data, corpus_info, vectors = _load_run(run_dir)
    if corpus_info["corpus_name"] != "memories":
        raise ConfigError("check the store on a --corpus memories run")
    memories = load_memories()
    if [m.memory_id for m in memories] != [d.doc_id for d in data.docs]:
        raise ConfigError("memories.jsonl order differs from the run's documents")
    results = {}
    for model, (query_vecs, doc_vecs) in vectors.items():
        dim = doc_vecs.shape[1]
        exact = ExactStore(dim)
        exact.upsert(memories, doc_vecs)
        folder = Path(tempfile.mkdtemp(prefix="memrag_chroma_"))
        try:
            chroma = ChromaStore(model, dim, folder)
            start = time.perf_counter()
            chroma.upsert(memories, doc_vecs)
            build_s = time.perf_counter() - start
            chroma.upsert(memories[:50], doc_vecs[:50])  # idempotent: must not add rows
            idempotent = chroma.count() == len(memories)
            row = compare(memories, data.queries, query_vecs, exact, chroma)
            disk = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            chroma.close()
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        results[model] = {"dim": dim, "memories": len(memories), "build_s": round(build_s, 3),
                          "disk_mib": round(disk / 2**20, 2), "idempotent_upsert": idempotent, **row}
    out = new_run_dir(bootstrap.project_root(), "store_check")
    write_json(out / "metrics.json", {"models": results})
    write_json(out / "manifest.json", {
        "run_id": out.name, "kind": "store_check", "source_run": source.name, "note": note,
        "created": datetime.now().isoformat(timespec="seconds"), **corpus_info, "queries": len(data.queries)})
    return {"run_dir": str(out), "models": results}


def rescore_retrieval(run_dir: str, note: str = "") -> dict:
    """Score a finished evaluation again with the current labels, reusing its saved vectors.

    No model is loaded, so it runs anywhere (also on Windows). Refuses when the corpus or
    the queries changed since the run. Writes a new results/runs/<run_id>_retrieval_rescore/.
    """
    from memrag.application.evaluation.retrieval_eval import PRIMARY_SCOPE, rank
    from memrag.infrastructure.storage.manifest import append_jsonl, new_run_dir, write_json

    source, old, data, corpus_info, vectors = _load_run(run_dir)
    out = new_run_dir(bootstrap.project_root(), "retrieval_rescore")
    per_model = {}
    for model, (query_vecs, doc_vecs) in vectors.items():
        rows = rank(query_vecs, doc_vecs, data)
        for row in rows:
            append_jsonl(out / "per_query.jsonl", {"model": model, **row})
        per_model[model] = rows
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


PROFILE_STORES = ("chroma", "numpy")


def _profile_setup(model: str, kind: str, stores: Sequence[str], fake: bool, note: str, settings: dict):
    """Run folder, manifest and the embedding config the child processes use."""
    from memrag.domain.errors import ConfigError, ModelNotAvailableError
    from memrag.infrastructure.storage.manifest import environment_info, new_run_dir, write_json

    bad = [s for s in stores if s not in PROFILE_STORES]
    if bad:
        raise ConfigError(f"stores must be in {PROFILE_STORES}, got {bad}")
    root = bootstrap.project_root()
    reg = bootstrap.registry(root)
    spec = reg.get(model)
    path = reg.local_path(spec)
    if path is None and not fake:
        raise ModelNotAvailableError(f"model {model!r} is not downloaded; run `memrag models download --models {model}`")
    run_dir = new_run_dir(root, kind + ("_fake" if fake else ""))
    manifest = {"run_id": run_dir.name, "kind": kind, "started": datetime.now().isoformat(timespec="seconds"),
                "note": note, "embedding_model": {"name": spec.name, "hf_id": spec.hf_id, "revision": spec.revision,
                                                  "dim": spec.dim, "query_prefix": spec.query_prefix},
                "stores": list(stores), **settings, "fake": fake, "environment": environment_info(root)}
    write_json(run_dir / "manifest.json", manifest)
    emb = {"spec": asdict(spec), "path": str(path) if path else None, "fake": fake}
    return run_dir, manifest, emb


def _write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def _profile_row(result: dict, **keys) -> dict:
    """A child result without its raw samples and hits, for the summary table."""
    return {**keys, **{k: v for k, v in result.items() if k not in ("samples_ms", "hits", "matched")}}


def profile_retrieval(model: str, devices: Sequence[str] = ("cpu", "cuda"), stores: Sequence[str] = PROFILE_STORES,
                      ks: Sequence[int] = (3, 5), repeats: int = 3, split: str | None = None, fake: bool = False,
                      note: str = "") -> dict:
    """Plan 5.1.7 experiment A: question → Top-k on the frozen memories, per device × store × Top-k.

    One child process per device and store; every query uses the pipeline filter (person and
    the query's dates, with the 5.1.6 settings). Recall@k against the labels, overlap with
    exact search, and the context each Top-k would send to the LLM.
    """
    import shutil

    from memrag.application.answer import build_context
    from memrag.application.evaluation.retrieval_profile import label_recall, overlap_recall, summarize_latency
    from memrag.domain.schemas.retrieval import MemoryFilter
    from memrag.infrastructure.retrieval_probe import run_job
    from memrag.infrastructure.storage.manifest import append_jsonl, write_json

    cfg = bootstrap.pipeline_config()
    r_cfg = cfg["retrieval"]
    data, corpus_info = _retrieval_input("memories", split)
    memories = load_memories()
    by_id = {m.memory_id: m for m in memories}
    settings = {"experiment": "A", "devices": list(devices), "ks": list(ks), "repeats": repeats,
                "queries": len(data.queries), "memories": len(memories), **corpus_info,
                "filter": {"date_fallback": r_cfg["date_fallback"], "category_filter": r_cfg["category_filter"]},
                "context_budget_chars": cfg["answer"]["context_budget_chars"]}
    run_dir, manifest, emb = _profile_setup(model, "profile_retrieval", stores, fake, note, settings)
    work = run_dir / "work"
    _write_records(work / "records.jsonl", memories)
    queries = []
    for q in data.queries:
        f = q.filters
        flt = MemoryFilter(person_id=q.person_id, category=f.category if r_cfg["category_filter"] else None,
                           date_from=f.date_from, date_to=f.date_to, date_fallback=r_cfg["date_fallback"])
        queries.append({"id": q.query_id, "text": q.text, "filters": {"pipeline": flt.model_dump(mode="json")}})

    results: dict[tuple[str, str], dict] = {}
    for device in devices:
        for store in stores:
            job = {"store": store, "model": model, "records": str(work / "records.jsonl"), "vectors": None,
                   "queries": queries, "query_vectors": None, "embedding": {**emb, "device": device},
                   "ks": list(ks), "repeats": repeats, "warmup": 5, "work_dir": str(work / f"{device}_{store}")}
            res = run_job(job, work / f"{device}_{store}")
            results[(device, store)] = res
            append_jsonl(run_dir / "raw.jsonl", {"device": device, "store": store, **res})

    rows = []
    budget = cfg["answer"]["context_budget_chars"]
    for (device, store), res in results.items():
        row = _profile_row(res, device=device, store=store)
        if res.get("ok"):
            hits = res["hits"]["pipeline"]
            exact = results.get((device, "numpy"), {})
            row["latency"] = summarize_latency(res["samples_ms"]["pipeline"])
            row["by_k"] = {}
            for k in ks:
                k_hits = hits[str(k)]
                ctx = [build_context([(by_id[m], -rank) for rank, m in enumerate(ids)], budget) for ids in k_hits.values()]
                row["by_k"][str(k)] = {
                    **label_recall(k_hits, data.grades, k),
                    "overlap_with_exact": overlap_recall(k_hits, exact["hits"]["pipeline"][str(k)])
                    if exact.get("ok") else None,
                    "context_chars_mean": round(sum(sum(len(line) + 1 for line in c.lines) for c in ctx) / len(ctx), 1),
                    "context_memories_mean": round(sum(len(c.items) for c in ctx) / len(ctx), 2),
                }
        rows.append(row)
    write_json(run_dir / "metrics.json", {"rows": rows})
    shutil.rmtree(work, ignore_errors=True)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "rows": rows}


def profile_scale(model: str, sizes: Sequence[int] = (1000, 10000, 50000), stores: Sequence[str] = PROFILE_STORES,
                  ks: Sequence[int] = (3, 5), repeats: int = 3, device: str = "cpu", fake: bool = False,
                  note: str = "") -> dict:
    """Plan 5.1.7 experiment B: filter cost on corpora expanded to each size, per store and scenario.

    Vectors of the frozen memories and the queries are computed once in a child process;
    each size × store is measured in its own child process without the embedding model.
    """
    import gc
    import shutil

    import numpy as np

    from memrag.application.evaluation.retrieval_profile import (
        SCALE_SCENARIOS,
        USERS,
        expand_corpus,
        overlap_recall,
        scale_queries,
        selectivity,
        summarize_latency,
        user_spans,
    )
    from memrag.domain.errors import PipelineError
    from memrag.infrastructure.retrieval_probe import run_job
    from memrag.infrastructure.storage.manifest import append_jsonl, write_json

    data, corpus_info = _retrieval_input("memories", None)
    memories = load_memories()
    settings = {"experiment": "B", "sizes": list(sizes), "users": USERS, "ks": list(ks), "repeats": repeats,
                "scenarios": list(SCALE_SCENARIOS), "queries": len(data.queries), "base_memories": len(memories),
                "vector_device": device, "noise_cosine": 0.98, **corpus_info}
    run_dir, manifest, emb = _profile_setup(model, "profile_scale", stores, fake, note, settings)
    work = run_dir / "work"
    work.mkdir(parents=True)
    embed = run_job({"mode": "embed", "embedding": {**emb, "device": device},
                     "documents": [m.content for m in memories], "queries": [q.text for q in data.queries],
                     "out_documents": str(work / "base_docs.npy"), "out_queries": str(work / "queries.npy")},
                    work / "embed")
    if not embed.get("ok"):
        raise PipelineError("embed", model, str(embed))
    base = np.load(work / "base_docs.npy")
    qids = [q.query_id for q in data.queries]

    rows = []
    for size in sizes:
        records, vectors = expand_corpus(memories, base, size)
        size_dir = work / str(size)
        _write_records(size_dir / "records.jsonl", records)
        np.save(size_dir / "vectors.npy", vectors)
        flts = scale_queries(qids, user_spans(records))
        queries = [{"id": qid, "text": None, "filters": {s: f[s].model_dump(mode="json") for s in SCALE_SCENARIOS}}
                   for qid, f in zip(qids, flts)]
        total = len(records)
        del records, vectors
        gc.collect()
        results = {}
        for store in stores:
            job = {"store": store, "model": model, "records": str(size_dir / "records.jsonl"),
                   "vectors": str(size_dir / "vectors.npy"), "queries": queries,
                   "query_vectors": str(work / "queries.npy"), "embedding": None, "ks": list(ks),
                   "repeats": repeats, "warmup": 5, "work_dir": str(size_dir / store)}
            results[store] = run_job(job, size_dir / store)
            append_jsonl(run_dir / "raw.jsonl", {"size": size, "store": store, **results[store]})
        exact = results.get("numpy", {})
        for store, res in results.items():
            row = _profile_row(res, size=size, store=store)
            if res.get("ok"):
                row["scenarios"] = {}
                for s in SCALE_SCENARIOS:
                    cell = {"latency": summarize_latency(res["samples_ms"][s])}
                    if exact.get("ok"):
                        cell["selectivity"] = selectivity(exact["matched"][s], total)
                        cell["overlap_with_exact"] = {str(k): overlap_recall(res["hits"][s][str(k)],
                                                                             exact["hits"][s][str(k)]) for k in ks}
                    row["scenarios"][s] = cell
            rows.append(row)
        shutil.rmtree(size_dir, ignore_errors=True)
    write_json(run_dir / "metrics.json", {"rows": rows})
    shutil.rmtree(work, ignore_errors=True)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "rows": rows}
