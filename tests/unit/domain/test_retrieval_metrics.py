import math

import pytest

from memrag.domain.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k
from memrag.domain.metrics.statistics import auc, bootstrap_ci, paired_bootstrap_ci

GRADES = {"a": 2, "b": 1, "c": 2}
RANKED = ["x", "a", "y", "b", "c"]


def test_recall_hand_computed():
    assert recall_at_k(RANKED, GRADES, 1) == 0.0
    assert recall_at_k(RANKED, GRADES, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, GRADES, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, GRADES, 10) == 1.0


def test_mrr_hand_computed():
    assert mrr_at_k(RANKED, GRADES, 10) == 0.5
    assert mrr_at_k(RANKED, GRADES, 1) == 0.0


def test_ndcg_hand_computed():
    dcg = 2 / math.log2(3) + 1 / math.log2(5) + 2 / math.log2(6)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, GRADES, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "c", "b"], GRADES, 10) == pytest.approx(1.0)
    # Ideal list is cut at k too
    assert ndcg_at_k(["a"], GRADES, 1) == pytest.approx(1.0)


def test_metrics_reject_query_without_relevant():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, {}, 5)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, {}, 5)


def test_auc():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5


def test_bootstrap_intervals():
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert low < 0.5 < high
    assert bootstrap_ci([0.3] * 10) == pytest.approx((0.3, 0.3))
    # a is always 0.1 higher than b: the difference interval is exactly 0.1
    a = [0.5, 0.7, 0.9, 0.6]
    assert paired_bootstrap_ci(a, [x - 0.1 for x in a]) == pytest.approx((0.1, 0.1))
