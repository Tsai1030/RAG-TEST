from datetime import date

import numpy as np
import pytest

from memrag.application.evaluation.dataset_check import GoldSet
from memrag.application.evaluation.retrieval_eval import build_input, candidate_mask, compare, rank, summarize
from memrag.domain.schemas.evaluation import GoldFact, MemoryLabel, Persona, QueryFilter, QueryLabel, TestQuery
from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord


def _fact(fid, pid, category="event", event_date=None, mention=date(2026, 10, 5)):
    return GoldFact(fact_id=fid, person_id=pid, category=category, statement="〇〇さんは散歩した。",
                    event_date=event_date, mention_date=mention)


def _query(qid, pid, qtype="direct", **filters):
    return TestQuery(query_id=qid, person_id=pid, text="散歩したのはいつ？", query_type=qtype,
                     filters=QueryFilter(**filters), asked_on=date(2026, 11, 1), split="dev")


FACTS = [
    _fact("f1", "p01", event_date=date(2026, 10, 3)),
    _fact("f2", "p01", category="plan", event_date=date(2026, 11, 10)),
    _fact("f3", "p01", category="preference"),
    _fact("f4", "p02", event_date=date(2026, 10, 3)),
    _fact("f5", "p01", mention=date(2026, 11, 2)),  # mentioned after the question
]


def _gold(queries=None):
    personas = [Persona(person_id=p, name="テスト", age=70, household="-", speech_style="-", profile="-")
                for p in ("p01", "p02")]
    queries = queries or [_query("q1", "p01"), _query("q2", "p01", "no_answer")]
    labels = [QueryLabel(query_id="q1", fact_id="f1", grade=2, annotator="t"),
              QueryLabel(query_id="q1", fact_id="f3", grade=1, annotator="t")]
    return GoldSet(personas=personas, facts=FACTS, queries=queries, labels=labels)


DOCS = build_input(_gold(), None).docs


def test_person_scope_never_crosses_people():
    q = _query("q1", "p01")
    for scope in ("person", "person+filter"):
        mask = candidate_mask(DOCS, q, scope)
        assert all(DOCS[i].person_id == "p01" for i in np.flatnonzero(mask))
    assert candidate_mask(DOCS, q, "all").tolist() == [True, True, True, True, False]


def test_date_and_category_filters():
    q = _query("q1", "p01", "time_category", date_from=date(2026, 11, 1), date_to=date(2026, 11, 30))
    # f3 has no event date, so a date filter drops it
    assert candidate_mask(DOCS, q, "person+filter").tolist() == [False, True, False, False, False]
    assert candidate_mask(DOCS, q, "person").tolist() == [True, True, True, False, False]
    q = _query("q2", "p01", "time_category", category="preference")
    assert candidate_mask(DOCS, q, "person+filter").tolist() == [False, False, True, False, False]


def test_rank_and_summarize():
    data = build_input(_gold(), None)
    # f4 (other person) is the closest overall, then f1, f3
    docs = np.array([[0.9, 0.1], [0.0, 1.0], [0.6, 0.8], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    rows = rank(queries, docs, data)
    by = {(r["query_id"], r["scope"]): r for r in rows}
    assert [t[0] for t in by["q1", "all"]["top"]] == ["f4", "f1", "f3", "f2"]
    assert by["q1", "all"]["top1_other_person"] is True
    assert by["q1", "person"]["ndcg@10"] == pytest.approx(1.0)
    assert by["q1", "all"]["recall@1"] == 0.0
    assert "ndcg@10" not in by["q2", "person"]
    s = summarize(rows)
    assert s["person"]["queries"] == 1
    assert s["person"]["no_answer_auc"] == 1.0  # q1 top-1 0.99 vs q2 top-1 -0.11
    assert s["all"]["top1_other_person_rate"] == 1.0


def _memory(content, facts, category="event", event_date=None, day=5):
    draft = MemoryDraft(category=category, content=content, event_date=event_date, source_turn_ids=[1])
    rec = MemoryRecord.from_draft(draft, person_id="p01", conversation_date=date(2026, 10, day), source_log_id="d")
    return rec.model_copy(update={"gold_fact_ids": facts})


def test_memory_corpus_inherits_best_fact_grade():
    mems = [_memory("佐藤さんは散歩した。", ["f1", "f3"]), _memory("佐藤さんは本が好き。", ["f3"]),
            _memory("天気がいい。", [])]
    data = build_input(_gold(), None, mems)
    g = data.grades["q1"]
    assert sorted(g.values()) == [1, 2] and mems[2].memory_id not in g
    assert data.unanswerable_in_corpus == []
    # q1's facts were never summarized: the query is reported and skipped
    data = build_input(_gold(), None, [mems[2]])
    assert data.unanswerable_in_corpus == ["q1"]
    vecs = np.ones((1, 2), dtype=np.float32) / np.sqrt(2)
    rows = rank(np.ones((2, 2), dtype=np.float32), vecs, data)
    assert {r["query_id"] for r in rows} == {"q2"}


def test_pooled_labels_override_inherited_grades():
    mems = [_memory("佐藤さんは散歩した。", ["f1"]), _memory("佐藤さんは歩くのが好き。", [])]
    labels = [MemoryLabel(query_id="q1", memory_id=mems[0].memory_id, grade=1, annotator="t"),
              MemoryLabel(query_id="q1", memory_id=mems[1].memory_id, grade=2, annotator="t")]
    g = build_input(_gold(), None, mems, labels).grades["q1"]
    assert g == {mems[0].memory_id: 1, mems[1].memory_id: 2}
    # A judged 0 removes an inherited grade; the query then has nothing relevant left
    zero = [MemoryLabel(query_id="q1", memory_id=mems[0].memory_id, grade=0, annotator="t")]
    data = build_input(_gold(), None, mems, zero)
    assert data.grades["q1"] == {} and data.unanswerable_in_corpus == ["q1"]


def test_relevant_memory_filtered_out_scores_zero():
    q = _query("q1", "p01", "time_category", category="event")
    mems = [_memory("佐藤さんは散歩した。", ["f1"], category="routine"), _memory("佐藤さんは料理した。", [])]
    data = build_input(_gold([q]), None, mems)
    vecs = np.eye(2, dtype=np.float32)
    rows = {r["scope"]: r for r in rank(np.array([[1.0, 0.0]], dtype=np.float32), vecs, data)}
    assert rows["person"]["ndcg@10"] == pytest.approx(1.0)
    assert rows["person+filter"]["ndcg@10"] == 0.0 and rows["person+filter"]["relevant_filtered_out"]


def test_compare_marks_best():
    data = build_input(_gold(), None)
    docs = np.eye(5, 2, dtype=np.float32) + 0.01
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    rows = rank(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), docs, data)
    out = compare({"a": rows, "b": rows})
    assert out["person"]["vs_best"]["a"]["diff"] == 0.0
    assert not out["person"]["vs_best"]["b"]["significant"]


def test_rank_rejects_mismatched_vectors():
    with pytest.raises(ValueError):
        rank(np.zeros((1, 2)), np.zeros((5, 2)), build_input(_gold(), None))
