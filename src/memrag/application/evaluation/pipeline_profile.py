"""Plan 5.1.7 full pipeline profile: per-stage latency, LLM token speed, profiler overhead.

Timings come from the pipeline's own stage laps (`ask`, `ingest_log`), a wall clock around
each call, and llama-server's per-request timings (`call_stats`); this module only turns
them into summaries.

- TTFT is server-side: prompt processing (prefill) plus one decoded token. Answers are not
  streamed, so the client sees the first token only with the whole answer.
- Coverage: how far the sum of the stage laps is from the wall clock around the call
  (acceptance: < 5%).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean

from memrag.application.evaluation.retrieval_profile import percentiles

ANSWER_STAGES = ("embed_query", "retrieve", "context", "prompt", "llm", "parse")
INGEST_STAGES = ("summarize", "embed", "store")
LLM_REASONS = ("answered", "llm_refused", "invalid_output")  # `ask` called the LLM


def coverage_gap(rows: Sequence[Mapping], stages: Sequence[str]) -> dict | None:
    """|wall − sum of stages| / wall over the rows (each row: timings_ms and wall_ms)."""
    gaps = [abs(r["wall_ms"] - sum(r["timings_ms"].get(s, 0.0) for s in stages)) / r["wall_ms"]
            for r in rows if r["wall_ms"] > 0]
    return {"mean": round(mean(gaps), 4), "max": round(max(gaps), 4)} if gaps else None


def stage_summary(rows: Sequence[Mapping], stages: Sequence[str]) -> dict:
    """Percentiles per stage (over the rows that ran it), of the wall clock, each stage's share
    of the summed wall time, and the coverage gap."""
    wall = sum(r["wall_ms"] for r in rows)
    out = {s: percentiles([r["timings_ms"][s] for r in rows if s in r["timings_ms"]]) for s in stages}
    out["wall"] = percentiles([r["wall_ms"] for r in rows])
    out["share"] = {s: round(sum(r["timings_ms"].get(s, 0.0) for r in rows) / wall, 4) if wall else None
                    for s in stages}
    out["coverage_gap"] = coverage_gap(rows, stages)
    return out


def llm_speed(calls: Sequence[Mapping]) -> dict:
    """Prefill and decode speed over llama-server calls; calls without server timings only count."""
    ok = [c for c in calls if c.get("prefill_ms") is not None and c.get("decode_tokens")]
    out: dict = {"calls": len(calls), "with_timings": len(ok), "retries": sum(c.get("retries", 0) for c in calls),
                 "wall_ms": percentiles([c["wall_ms"] for c in calls])}
    if not ok:
        return out
    prefill_tokens = sum(c["prefill_tokens"] or 0 for c in ok)
    prefill_ms = sum(c["prefill_ms"] for c in ok)
    return {**out,
            "ttft_ms": percentiles([c["prefill_ms"] + c["decode_ms"] / c["decode_tokens"] for c in ok]),
            "prefill_ms": percentiles([c["prefill_ms"] for c in ok]),
            "decode_ms": percentiles([c["decode_ms"] for c in ok]),
            "prompt_tokens": percentiles([c["prompt_tokens"] for c in ok if c.get("prompt_tokens") is not None]),
            "cached_tokens": percentiles([c["cached_tokens"] for c in ok if c.get("cached_tokens") is not None]),
            "completion_tokens": percentiles([c["decode_tokens"] for c in ok]),
            "prefill_tok_s": round(prefill_tokens / prefill_ms * 1000, 1) if prefill_ms else None,
            "decode_tok_s": round(sum(c["decode_tokens"] for c in ok) / sum(c["decode_ms"] for c in ok) * 1000, 1),
            "decode_tok_s_per_call": percentiles([c["decode_tokens"] / c["decode_ms"] * 1000 for c in ok]),
            # HTTP, JSON and grammar setup outside the server's prefill and decode
            "overhead_ms": percentiles([c["wall_ms"] - c["prefill_ms"] - c["decode_ms"] for c in ok])}


def answer_summary(rows: Sequence[Mapping]) -> dict:
    """Rows of one Top-k: outcomes, stages with and without the LLM call, LLM speed, context size."""
    with_llm = [r for r in rows if r["reason"] in LLM_REASONS]
    without = [r for r in rows if r["reason"] not in LLM_REASONS]
    return {"queries": len(rows), "reasons": dict(Counter(r["reason"] for r in rows)),
            "with_llm": stage_summary(with_llm, ANSWER_STAGES) if with_llm else None,
            "without_llm": stage_summary(without, ANSWER_STAGES) if without else None,
            "llm": llm_speed([r["llm"] for r in with_llm if r.get("llm")]),
            "context_chars": percentiles([r["context_chars"] for r in with_llm]),
            "coverage_gap": coverage_gap(rows, ANSWER_STAGES)}


def ingest_summary(rows: Sequence[Mapping]) -> dict:
    """Rows of ingested logs: success, retries, stages, LLM speed, and the summarize time outside the LLM
    (prompt rendering and validation)."""
    calls = [c for r in rows for c in r["llm_calls"]]
    outside = [r["timings_ms"]["summarize"] - sum(c["wall_ms"] for c in r["llm_calls"]) for r in rows]
    return {"logs": len(rows), "ok": sum(r["ok"] for r in rows), "memories": sum(r["accepted"] for r in rows),
            "retries": sum(r["retries"] for r in rows), "llm_calls": len(calls),
            "stages": stage_summary(rows, INGEST_STAGES), "llm": llm_speed(calls),
            "summarize_outside_llm_ms": percentiles(outside),
            "memories_per_log": round(mean(r["accepted"] for r in rows), 2) if rows else None}


def lap_overhead_us(n: int = 100_000) -> float:
    """Cost of one stage lap as `ask` takes it (clock read, rounding, dict write), in microseconds."""
    timings: dict[str, float] = {}
    start = clock = time.perf_counter()
    for _ in range(n):
        now = time.perf_counter()
        timings["stage"] = round((now - clock) * 1000, 2)
        clock = now
    return round((time.perf_counter() - start) / n * 1e6, 3)


def sampler_overhead(on_ms: Sequence[float], off_ms: Sequence[float]) -> dict:
    """The same retrieval passes with and without the memory sampler thread running."""
    on, off = percentiles(on_ms), percentiles(off_ms)
    return {"on": on, "off": off,
            "p50_relative": round(on["p50"] / off["p50"] - 1, 4) if off.get("p50") else None,
            "mean_relative": round(on["mean"] / off["mean"] - 1, 4) if off.get("mean") else None}
