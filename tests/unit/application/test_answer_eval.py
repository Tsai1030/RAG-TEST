from datetime import date

from memrag.application.answer import Answer
from memrag.application.evaluation.answer_eval import expected, score_answer, summarize
from memrag.domain.schemas.evaluation import TestQuery
from memrag.domain.schemas.memory import MemoryDraft, MemoryRecord
from memrag.domain.schemas.retrieval import MemoryFilter


def _rec(person, content):
    d = MemoryDraft(category="event", content=content, source_turn_ids=[1])
    return MemoryRecord.from_draft(d, person_id=person, conversation_date=date(2026, 10, 5), source_log_id="d")


MINE, THEIRS = _rec("p01", "佐藤さんは散歩した。"), _rec("p02", "鈴木さんは将棋をした。")
RECORDS = {MINE.memory_id: MINE, THEIRS.memory_id: THEIRS}
Q = TestQuery(query_id="q1", person_id="p01", text="散歩したっけ？", query_type="direct", asked_on=date(2026, 11, 1),
              split="dev")


def _answer(refused, citations, reason=None):
    return Answer("散歩したっけ？", "x", refused, reason or ("llm_refused" if refused else "answered"), citations, [],
                  MemoryFilter(person_id="p01"), [], [], {}, 0.8)


def test_expected():
    assert expected({"m": 2, "n": 1}) == "answer"
    assert expected({"m": 1}) == "partial"
    assert expected({}) == "refuse"


def test_score_and_summarize():
    rows = [
        score_answer(Q, _answer(False, [MINE.memory_id]), {MINE.memory_id: 2}, RECORDS,
                     ({MINE.memory_id}, {"m_old"})),
        score_answer(Q, _answer(True, [], "below_threshold"), {MINE.memory_id: 2}, RECORDS),
        score_answer(Q, _answer(True, []), {}, RECORDS),
        score_answer(Q, _answer(False, [THEIRS.memory_id]), {}, RECORDS),  # cites another user's memory
    ]
    assert rows[3]["outside_filter"] == [THEIRS.memory_id]
    m = summarize(rows)
    assert m["answerable"]["answered"]["rate"] == 0.5 and m["answerable"]["cited_best"]["rate"] == 1.0
    assert m["answerable"]["refused_by"] == {"below_threshold": 1}
    assert m["no_answer"]["refused"]["rate"] == 0.5
    assert m["citations"] == {"invalid": 0, "outside_filter": 1, "answered_without_citation": 0}
    assert m["updates"] == {"queries": 1, "answered": 1, "cited_new": 1, "cited_old_only": 0}
    assert m["partial"]["answered"] == {"n": 0}
