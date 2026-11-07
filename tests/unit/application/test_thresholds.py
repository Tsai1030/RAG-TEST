from datetime import date

import numpy as np
import pytest

from memrag.application.evaluation import thresholds as th
from memrag.application.evaluation.retrieval_eval import Doc, EvalInput
from memrag.domain.metrics.statistics import cluster_bootstrap_rate
from memrag.domain.schemas.evaluation import TestQuery


def _doc(i, pid="p01"):
    return Doc(f"m{i}", pid, "x", "event", None, date(2026, 10, 5))


def _query(qid, pid="p01"):
    return TestQuery(query_id=qid, person_id=pid, text="散歩したのはいつ？", query_type="direct",
                     asked_on=date(2026, 11, 1), split="dev")


# p01 has m0 (answer, grade 2), m1 (grade 1) and m2..m6 (grade 0); m7 belongs to p02
DATA = EvalInput(
    queries=[_query("q1"), _query("q2")],
    docs=[_doc(i) for i in range(7)] + [_doc(7, "p02")],
    grades={"q1": {"m0": 2, "m1": 1}, "q2": {}},
    unanswerable_in_corpus=[],
)


def test_candidate_sets_shape():
    sets = th.candidate_sets(DATA, sizes=(2, 3), samples=4)
    assert sets == th.candidate_sets(DATA, sizes=(2, 3), samples=4)  # deterministic
    for s in sets:
        assert 7 not in s.doc_indices  # never another user's memory
        if s.query_index == 0:  # for q1, m1 is grade 1 and never sampled; m0 only in positive sets
            assert 1 not in s.doc_indices
            assert (0 in s.doc_indices) == s.positive
    assert sum(s.positive for s in sets) == 8  # only q1 has an answer: 2 sizes x 4 samples
    assert sum(not s.positive for s in sets) == 16  # q1 without its answer, and q2


def test_calibrate_and_evaluate():
    # query vectors are unit x; doc score = its x component
    xs = [0.9, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.99]
    docs = np.array([[x, np.sqrt(1 - x * x)] for x in xs])
    queries = np.array([[1.0, 0.0]])
    data = EvalInput(DATA.queries[:1], DATA.docs, {"q1": DATA.grades["q1"]}, [])
    sets = th.candidate_sets(data, sizes=(2, 3), samples=5)
    t = th.calibrate(queries, docs, sets, target=0.0)
    assert t == pytest.approx(0.6001)  # just above the best grade-0 score
    out = th.evaluate(queries, docs, data, sets, t)["by_size"]["all"]
    assert out["false_recall"]["rate"] == 0.0
    assert out["false_block"]["rate"] == 0.0  # the answer (0.9) always passes
    assert out["top1_wrong"]["rate"] == 0.0
    assert out["wrong_pass_rate"] == 0.0
    # A threshold above the answer blocks it every time
    high = th.evaluate(queries, docs, data, sets, 0.95)["by_size"]["all"]
    assert high["false_block"]["rate"] == 1.0


def test_cluster_bootstrap_rate():
    assert cluster_bootstrap_rate([[True, True], [True]]) == (1.0, 1.0)
    low, high = cluster_bootstrap_rate([[True] * 10, [False] * 10])
    assert low == 0.0 and high == 1.0  # two queries: the interval spans everything
    with pytest.raises(ValueError):
        cluster_bootstrap_rate([])
