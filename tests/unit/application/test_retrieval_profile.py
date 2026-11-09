"""Plan 5.1.7 retrieval and scale test: expanded corpus, filter scenarios and summaries."""

from datetime import date

import numpy as np
import pytest

from memrag.application.evaluation.retrieval_profile import (
    SCALE_SCENARIOS,
    expand_corpus,
    label_recall,
    overlap_recall,
    percentiles,
    scale_queries,
    selectivity,
    summarize_latency,
    user_spans,
)
from tests.unit.infrastructure.test_numpy_store import _corpus


def test_expand_corpus():
    records, vecs = _corpus(n=40, dim=64)
    out, noisy = expand_corpus(records, vecs, total=500, users=5)
    assert len(out) == 500 and noisy.shape == (500, 64)
    assert len({r.memory_id for r in out}) == 500
    assert sorted({r.person_id for r in out}) == ["v00", "v01", "v02", "v03", "v04"]
    assert np.allclose(np.linalg.norm(noisy, axis=1), 1, atol=1e-5)
    # Each copy stays close to its source vector but is never identical to it
    src = {r.content: v for r, v in zip(records, vecs)}
    cos = np.array([float(noisy[i] @ src[r.content]) for i, r in enumerate(out)])
    assert 0.95 < cos.mean() < 0.995 and cos.max() < 0.9999
    # An event keeps its distance to the day it was told; dates end at the fixed day
    by_content = {r.content: r for r in records}
    for r in out:
        base = by_content[r.content]
        if base.event_date:
            assert r.event_date - r.conversation_date == base.event_date - base.conversation_date
            assert r.event_date_ord == int(r.event_date.strftime("%Y%m%d"))
        else:
            assert r.event_date is None and r.event_date_ord is None
    assert max(r.conversation_date for r in out) <= date(2026, 10, 31)
    # Same seed, same corpus
    again, again_vecs = expand_corpus(records, vecs, total=500, users=5)
    assert again == out and np.array_equal(again_vecs, noisy)


def test_span_grows_with_memories_per_user():
    records, vecs = _corpus(n=40, dim=8)
    small, _ = expand_corpus(records, vecs, total=200, users=2)
    large, _ = expand_corpus(records, vecs, total=5000, users=2)
    def span(rs):
        return (max(r.conversation_date for r in rs) - min(r.conversation_date for r in rs)).days

    assert span(small) < 61 and span(large) > 400


def test_scale_queries():
    records, vecs = _corpus(n=40, dim=8)
    out, _ = expand_corpus(records, vecs, total=300, users=3)
    flts = scale_queries([f"q{i}" for i in range(9)], user_spans(out))
    assert len(flts) == 9 and all(tuple(f) == SCALE_SCENARIOS for f in flts)
    assert [f["person"].person_id for f in flts[:4]] == ["v00", "v01", "v02", "v00"]
    f = flts[0]
    assert f["none"].person_id is None and f["category"].person_id is None
    assert (f["person+week"].date_to - f["person+week"].date_from).days == 6
    assert not f["person+week"].date_fallback and f["person+week+fallback"].date_fallback
    assert (f["person+month+fallback"].date_to - f["person+month+fallback"].date_from).days == 29
    assert len(set(f["person+3categories"].categories)) == 3
    assert f["person+category"].category == f["category"].category == f["person+3categories"].categories[0]


def test_summaries():
    assert percentiles([]) == {"n": 0}
    p = percentiles(list(range(1, 101)))
    assert (p["p50"], p["p95"], p["p99"], p["max"]) == (50, 95, 99, 100)
    lat = summarize_latency({"5": {"embed": [10, 10], "search": [2, 4], "get": [1, 1]}})["5"]
    assert lat["total"]["mean"] == 14 and lat["qps"] == pytest.approx(1000 / 14, abs=0.01)
    no_embed = summarize_latency({"3": {"embed": [None, None], "search": [1, 3], "get": [0, 0]}})["3"]
    assert no_embed["embed"] is None and no_embed["total"]["mean"] == 2
    assert overlap_recall({"q1": ["a", "b"], "q2": ["c"]}, {"q1": ["a", "x"], "q2": ["c"], "q3": []}) == 0.75
    grades = {"q1": {"a": 2, "b": 1}, "q2": {"c": 1}, "q3": {}}
    r = label_recall({"q1": ["b", "z", "a"], "q2": ["z"], "q3": ["a"]}, grades, k=2)
    assert r == {"queries": 2, "recall": 0.25, "best_in_top_k": 0.0}
    assert selectivity({"q1": 10, "q2": 30}, 100) == {"mean_matched": 20.0, "mean_share": 0.2,
                                                      "min_matched": 10, "max_matched": 30}
