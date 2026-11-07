import pytest

from memrag.application.evaluation import selection as sel

WEIGHTS = {
    "ndcg@10": {"weight": 0.6, "better": "higher"},
    "query_latency_p95_ms": {"weight": 0.4, "better": "lower"},
}
GATES = {"commercial_license": True, "person_filter_cross_user_errors": 0,
         "small_set_false_recall_ci_low_max": 0.05, "query_latency_p95_ms_max": 500, "memory_peak_gib_max": 3.0}


def _m(name, ndcg, latency, **kw):
    metrics = {"ndcg@10": ndcg, "recall@5": 0.9, "small_set_false_block": 0.3,
               "query_latency_p95_ms": latency, "memory_peak_mib": 900}
    return sel.ModelFacts(name, kw.get("role", "candidate"), kw.get("commercial_ok", True), metrics,
                          kw.get("cross", 0), kw.get("fr_low", 0.02))


def test_gates():
    models = [_m("ok", 0.9, 30), _m("nc", 0.9, 30, commercial_ok=False), _m("slow", 0.9, 600),
              _m("cross", 0.9, 30, cross=1), _m("recall", 0.9, 30, fr_low=0.06)]
    sel.apply_gates(models, GATES)
    assert [bool(m.gate_failures) for m in models] == [False, True, True, True, True]


def test_weighted_scores_min_max():
    models = [_m("a", 0.90, 40), _m("b", 0.80, 20), _m("c", 0.85, 30)]
    s = sel.weighted_scores(models, WEIGHTS)
    assert s["a"]["normalised"] == {"ndcg@10": 1.0, "query_latency_p95_ms": 0.0}
    assert s["b"]["normalised"] == {"ndcg@10": 0.0, "query_latency_p95_ms": 1.0}
    assert s["a"]["score"] == pytest.approx(0.6)
    assert s["b"]["score"] == pytest.approx(0.4)
    assert s["c"]["score"] == pytest.approx(0.5)


def test_sensitivity_reports_first_place_changes():
    models = [_m("a", 0.90, 40), _m("b", 0.80, 20)]
    rows = sel.sensitivity(models, WEIGHTS, delta=0.15)
    flips = {(r["metric"], r["change"]): r["first"] for r in rows}
    assert flips[("ndcg@10", "+0.15")] == "a"
    assert flips[("ndcg@10", "-0.15")] == "b"  # 0.45 vs 0.55: latency now wins
    assert any(not r["same"] for r in rows)
