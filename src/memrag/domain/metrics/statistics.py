"""Statistics on plain Python values (no numpy), so the domain stays framework-free."""

from __future__ import annotations

import random
from collections import Counter
from typing import Hashable, Sequence

BOOTSTRAP_SAMPLES = 2000


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Unweighted Cohen's kappa between two annotators' labels for the same items.

    Returns 1.0 when both annotators use a single identical label throughout
    (expected agreement is 1, so kappa is undefined; perfect agreement is reported).
    """
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no items to compare")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in count_a.keys() | count_b.keys()) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)


def bootstrap_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], samples: int = BOOTSTRAP_SAMPLES,
                        seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of mean(a - b) over the same items (queries)."""
    if len(a) != len(b):
        raise ValueError(f"paired lists differ in length: {len(a)} vs {len(b)}")
    return bootstrap_ci([x - y for x, y in zip(a, b)], samples, seed)


def cluster_bootstrap_rate(groups: Sequence[Sequence[bool]], samples: int = BOOTSTRAP_SAMPLES,
                           seed: int = 0) -> tuple[float, float]:
    """95% interval of a rate whose outcomes come in correlated groups (e.g. many sets per query).

    Whole groups are resampled, so the interval reflects the number of queries, not of sets.
    """
    if not groups or not any(groups):
        raise ValueError("no outcomes")
    rng = random.Random(seed)
    n = len(groups)
    rates = []
    for _ in range(samples):
        picked = [groups[rng.randrange(n)] for _ in range(n)]
        total = sum(len(g) for g in picked)
        rates.append(sum(sum(g) for g in picked) / total if total else 0.0)
    rates.sort()
    return rates[int(0.025 * samples)], rates[int(0.975 * samples) - 1]


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not positives or not negatives:
        raise ValueError("AUC needs at least one positive and one negative")
    wins = sum((p > q) + 0.5 * (p == q) for p in positives for q in negatives)
    return wins / (len(positives) * len(negatives))
