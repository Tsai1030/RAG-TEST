import pytest

from memrag.domain.metrics.statistics import cohen_kappa


def test_kappa_hand_computed():
    # 10 items: agree on 7. p_o = 0.7
    a = [2, 2, 1, 0, 0, 0, 0, 1, 2, 0]
    b = [2, 1, 1, 0, 0, 0, 2, 0, 2, 0]
    # counts a: 0->5, 1->2, 2->3 ; b: 0->5, 1->2, 2->3 ; p_e = (25+4+9)/100 = 0.38
    assert cohen_kappa(a, b) == pytest.approx((0.7 - 0.38) / (1 - 0.38))


def test_kappa_perfect_and_single_label():
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert cohen_kappa([0, 0, 0], [0, 0, 0]) == 1.0


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([0, 1], [0])
    with pytest.raises(ValueError):
        cohen_kappa([], [])
