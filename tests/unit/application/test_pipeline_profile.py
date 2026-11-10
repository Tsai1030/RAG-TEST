"""Plan 5.1.7 full pipeline profile: stage, LLM and overhead summaries."""

import pytest

from memrag.application.evaluation.pipeline_profile import (
    ANSWER_STAGES,
    answer_summary,
    coverage_gap,
    ingest_summary,
    lap_overhead_us,
    llm_speed,
    sampler_overhead,
    stage_summary,
)
from memrag.infrastructure.llm.llama_server import call_stats


def _call(prefill_ms=100.0, decode_tokens=20, decode_ms=400.0, wall_ms=520.0):
    return {"wall_ms": wall_ms, "retries": 0, "prompt_tokens": 300, "completion_tokens": decode_tokens,
            "cached_tokens": 250, "prefill_tokens": 50, "prefill_ms": prefill_ms, "decode_tokens": decode_tokens,
            "decode_ms": decode_ms}


def test_call_stats_reads_llama_server_timings():
    response = {"usage": {"prompt_tokens": 300, "completion_tokens": 20},
                "timings": {"cache_n": 250, "prompt_n": 50, "prompt_ms": 100.0, "predicted_n": 20, "predicted_ms": 400.0}}
    assert call_stats(response, 520.004, 1) == {**_call(), "retries": 1}
    assert call_stats({}, 5.0)["prefill_ms"] is None  # an older server without timings


def test_llm_speed():
    s = llm_speed([_call(), _call(prefill_ms=200.0, decode_tokens=40, decode_ms=800.0, wall_ms=1010.0)])
    assert s["calls"] == s["with_timings"] == 2
    assert s["ttft_ms"]["p50"] == pytest.approx(120.0)  # prefill + one token (400 / 20)
    assert s["decode_tok_s"] == 50.0 and s["prefill_tok_s"] == pytest.approx(333.3)
    assert s["overhead_ms"]["max"] == pytest.approx(20.0)
    assert llm_speed([{"wall_ms": 0.0, "retries": 0}]) == {"calls": 1, "with_timings": 0, "retries": 0,
                                                           "wall_ms": {"n": 1, "mean": 0.0, "p50": 0.0, "p95": 0.0,
                                                                       "p99": 0.0, "max": 0.0}}


def _row(reason="answered", llm=600.0, wall=None):
    t = {"embed_query": 30.0, "retrieve": 1.0, "context": 0.1}
    if reason == "answered":
        t |= {"prompt": 0.5, "llm": llm, "parse": 0.2}
    total = sum(t.values())
    return {"reason": reason, "timings_ms": {**t, "total": total}, "wall_ms": wall or total + 0.3,
            "context_chars": 400, "llm": _call(wall_ms=llm) if reason == "answered" else None}


def test_stage_summary_and_coverage():
    rows = [_row(), _row(llm=900.0)]
    s = stage_summary(rows, ANSWER_STAGES)
    assert s["llm"]["n"] == 2 and s["share"]["llm"] > 0.9
    assert s["coverage_gap"]["max"] < 0.01
    assert coverage_gap([_row(wall=1000.0)], ANSWER_STAGES)["max"] == pytest.approx(1 - 631.8 / 1000)


def test_answer_summary_splits_llm_calls():
    s = answer_summary([_row(), _row(), _row("below_threshold")])
    assert s["reasons"] == {"answered": 2, "below_threshold": 1}
    assert s["with_llm"]["llm"]["n"] == 2 and s["without_llm"]["llm"] == {"n": 0}
    assert s["llm"]["calls"] == 2 and s["context_chars"]["n"] == 2


def test_ingest_summary_counts_retries_and_time_outside_llm():
    rows = [{"ok": True, "accepted": 3, "retries": 1, "llm_calls": [_call(wall_ms=500.0), _call(wall_ms=400.0)],
             "timings_ms": {"summarize": 910.0, "embed": 40.0, "store": 2.0}, "wall_ms": 960.0},
            {"ok": False, "accepted": 0, "retries": 3, "llm_calls": [_call(wall_ms=100.0)] * 4,
             "timings_ms": {"summarize": 405.0}, "wall_ms": 406.0}]
    s = ingest_summary(rows)
    assert (s["logs"], s["ok"], s["memories"], s["retries"], s["llm_calls"]) == (2, 1, 3, 4, 6)
    assert s["summarize_outside_llm_ms"]["max"] == pytest.approx(10.0)
    assert s["stages"]["embed"]["n"] == 1  # a failed log is never embedded


def test_overheads():
    assert 0 < lap_overhead_us(1000) < 1000
    s = sampler_overhead([11.0, 11.0], [10.0, 10.0])
    assert s["p50_relative"] == pytest.approx(0.1)
    assert sampler_overhead([], [])["p50_relative"] is None
