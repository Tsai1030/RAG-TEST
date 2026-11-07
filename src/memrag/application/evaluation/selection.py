"""Two-stage model selection (plan 5.1.9) from the frozen rules in configs/selection.yaml.

Stage 1 drops models failing a hard gate. Stage 2 min-max normalises each weighted
metric across the remaining candidates (best = 1, worst = 0) and sums weight * value.
The sensitivity check moves each weight by +/- delta (renormalising the rest) and
reports whether the first place changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelFacts:
    """Everything the rules need about one model, gathered from the finished runs."""

    name: str
    role: str
    commercial_ok: bool
    metrics: dict[str, float]  # ndcg@10, recall@5, small_set_false_block, query_latency_p95_ms, memory_peak_mib
    cross_user_errors: int
    false_recall_ci_low: float
    gate_failures: list[str] = field(default_factory=list)


def apply_gates(models: list[ModelFacts], gates: dict) -> None:
    for m in models:
        fails = []
        if gates.get("commercial_license") and not m.commercial_ok:
            fails.append("授權不可商用")
        if m.cross_user_errors > gates.get("person_filter_cross_user_errors", 0):
            fails.append(f"只搜本人時出現別人的記憶 {m.cross_user_errors} 次")
        if m.false_recall_ci_low > gates["small_set_false_recall_ci_low_max"]:
            fails.append(f"小候選集無答案誤召回率的信賴區間下限 {m.false_recall_ci_low:.3f} 超過門檻")
        if m.metrics["query_latency_p95_ms"] > gates["query_latency_p95_ms_max"]:
            fails.append(f"延遲 p95 {m.metrics['query_latency_p95_ms']} ms 超過門檻")
        if m.metrics["memory_peak_mib"] > gates["memory_peak_gib_max"] * 1024:
            fails.append(f"記憶體峰值 {m.metrics['memory_peak_mib']} MiB 超過門檻")
        m.gate_failures = fails


def weighted_scores(models: list[ModelFacts], weights: dict) -> dict[str, dict]:
    """Per model: normalised value per metric and the total score."""
    out: dict[str, dict] = {m.name: {"normalised": {}, "score": 0.0} for m in models}
    for metric, rule in weights.items():
        values = {m.name: m.metrics[metric] for m in models}
        low, high = min(values.values()), max(values.values())
        for name, v in values.items():
            norm = 1.0 if high == low else (v - low) / (high - low)
            if rule["better"] == "lower":
                norm = 1.0 - norm if high != low else 1.0
            out[name]["normalised"][metric] = round(norm, 4)
            out[name]["score"] += rule["weight"] * norm
    for name in out:
        out[name]["score"] = round(out[name]["score"], 4)
    return out


def sensitivity(models: list[ModelFacts], weights: dict, delta: float = 0.05) -> list[dict]:
    """First place after moving each weight up or down by `delta` (the others rescaled to keep the sum)."""
    base = _winner(weighted_scores(models, weights))
    rows = []
    for metric in weights:
        for sign in (1, -1):
            new = max(0.0, weights[metric]["weight"] + sign * delta)
            rest = sum(w["weight"] for k, w in weights.items() if k != metric)
            scale = (1.0 - new) / rest if rest else 0.0
            changed = {k: {**w, "weight": new if k == metric else w["weight"] * scale} for k, w in weights.items()}
            winner = _winner(weighted_scores(models, changed))
            rows.append({"metric": metric, "change": f"{sign * delta:+.2f}", "first": winner, "same": winner == base})
    return rows


def _winner(scores: dict[str, dict]) -> str:
    return max(scores, key=lambda n: scores[n]["score"])
