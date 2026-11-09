"""Dialogue log -> memories in the vector store: summarize, embed, upsert (plan 5.1.6)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from memrag.application.summarize import summarize_log
from memrag.domain.ports import (
    EmbeddingProvider,
    LLMProvider,
    PromptStore,
    ResponseCache,
    VectorStore,
)
from memrag.domain.schemas.dialogue import DialogueLog


@dataclass
class IngestReport:
    log_id: str
    ok: bool
    accepted: int  # memories written to the store
    rejected: int  # LLM attempts that failed validation
    retries: int
    memory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # of the last attempt when not ok
    timings_ms: dict[str, float] = field(default_factory=dict)


def ingest_log(log: DialogueLog, user_name: str, embedder: EmbeddingProvider, store: VectorStore,
               llm: LLMProvider, prompts: PromptStore, cache: ResponseCache | None = None,
               version: str = "v1") -> IngestReport:
    """Nothing is written when the summary still fails validation after the retries."""
    start = time.perf_counter()
    result = summarize_log(log, user_name, llm, prompts, cache, version)
    summarize_ms = (time.perf_counter() - start) * 1000
    report = IngestReport(log.log_id, result.ok, 0, sum(bool(a.errors) for a in result.attempts),
                          max(len(result.attempts) - 1, 0), timings_ms={"summarize": round(summarize_ms, 2)})
    if not result.ok:
        report.errors = result.attempts[-1].errors if result.attempts else ["no attempt"]
        return report
    clock = time.perf_counter()
    vectors = embedder.embed_documents([r.content for r in result.records]) if result.records else []
    report.timings_ms["embed"] = round((time.perf_counter() - clock) * 1000, 2)
    clock = time.perf_counter()
    if result.records:
        store.upsert(result.records, vectors)
    report.timings_ms["store"] = round((time.perf_counter() - clock) * 1000, 2)
    report.accepted = len(result.records)
    report.memory_ids = [r.memory_id for r in result.records]
    return report
