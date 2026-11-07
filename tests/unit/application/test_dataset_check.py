from memrag.application.evaluation.dataset_check import GoldSet, agreement, assign_splits, check_gold_set, second_pass_sample
from memrag.domain.schemas.evaluation import GoldFact, Persona, QueryLabel, Split, TestQuery

PERSONA = dict(name="佐藤 和子", age=78, household="一人暮らし", speech_style="丁寧語", profile="バラが好き")


def fact(fid, pid="p01", **kw):
    values = dict(fact_id=fid, person_id=pid, category="plan", statement="佐藤さんは11月4日に健康診断を受ける予定だ。",
                  event_date="2026-11-04", mention_date="2026-10-20")
    return GoldFact(**(values | kw))


def query(qid, pid="p01", qtype="direct", **kw):
    values = dict(query_id=qid, person_id=pid, text="健診はいつだったかしら。", query_type=qtype,
                  asked_on="2026-11-01", split="test")
    return TestQuery(**(values | kw))


def label(qid, fid, grade=2, annotator="claude-1"):
    return QueryLabel(query_id=qid, fact_id=fid, grade=grade, annotator=annotator)


def gold(facts, queries, labels):
    personas = [Persona(person_id=p, **PERSONA) for p in ("p01", "p02")]
    return GoldSet(personas=personas, facts=facts, queries=queries, labels=labels)


def errors(gs):
    return [i.message for i in check_gold_set(gs).issues if i.level == "error"]


def test_clean_set_has_no_errors():
    assert errors(gold([fact("f1")], [query("q1")], [label("q1", "f1")])) == []


def test_label_on_other_persons_fact():
    gs = gold([fact("f1"), fact("f2", pid="p02", statement="鈴木さんは11月4日に健診を受ける予定だ。")],
              [query("q1")], [label("q1", "f2")])
    assert any("another person" in m for m in errors(gs))


def test_label_must_pass_query_filters():
    gs = gold([fact("f1")], [query("q1", qtype="time_category", filters={"category": "health"})], [label("q1", "f1")])
    assert any("does not pass" in m for m in errors(gs))
    gs = gold([fact("f1")], [query("q1", qtype="time_category", filters={"date_from": "2026-10-25", "date_to": "2026-10-31"})],
              [label("q1", "f1")])
    assert any("outside" in m for m in errors(gs))


def test_no_answer_rules():
    assert any("no-answer" in m for m in errors(gold([fact("f1")], [query("q1", qtype="no_answer")], [label("q1", "f1")])))
    assert any("no relevant fact" in m for m in errors(gold([fact("f1")], [query("q1")], [])))


def test_supersedes_must_point_to_earlier_fact_of_same_person():
    later = fact("f2", statement="佐藤さんは健康診断を11月5日に変更した。", mention_date="2026-10-25", supersedes="f1")
    assert errors(gold([fact("f1"), later], [query("q1")], [label("q1", "f2")])) == []
    early = fact("f2", statement="佐藤さんは健康診断を11月5日に変更した。", mention_date="2026-10-01", supersedes="f1")
    assert any("supersedes" in m for m in errors(gold([fact("f1"), early], [query("q1")], [label("q1", "f2")])))


def test_fact_mentioned_after_query_cannot_be_relevant():
    gs = gold([fact("f1", mention_date="2026-11-02")], [query("q1")], [label("q1", "f1")])
    assert any("after the query" in m for m in errors(gs))


def test_splits_are_stratified_and_stable():
    qs = [query(f"q{i}", qtype=t) for i, t in enumerate(["direct"] * 10 + ["no_answer"] * 5)]
    a, b = assign_splits(qs), assign_splits(list(reversed(qs)))
    assert a == b
    assert sum(1 for q in qs[:10] if a[q.query_id] == Split.DEV) == 4
    assert sum(1 for q in qs[10:] if a[q.query_id] == Split.DEV) == 2


def test_second_pass_sample_covers_every_type():
    qs = [query(f"q{i}", qtype=t) for i, t in enumerate(["direct"] * 8 + ["no_answer"] * 2)]
    sample = second_pass_sample(qs)
    assert len(sample) == 3  # 25% of 8 = 2, and at least 1 of 2
    assert any(qid in sample for qid in ("q8", "q9"))


def test_agreement_counts_unlisted_pairs_as_zero():
    facts = [fact("f1"), fact("f2", statement="佐藤さんはバラが好きだ。", category="preference", event_date=None)]
    labels = [label("q1", "f1", 2), label("q1", "f1", 2, "claude-2"), label("q1", "f2", 1, "claude-2")]
    out = agreement(gold(facts, [query("q1")], labels), "claude-1", "claude-2", ["q1"])
    assert out["pairs"] == 2
    assert out["exact_agreement"] == 0.5
    assert out["disagreements"] == [{"query_id": "q1", "fact_id": "f2", "claude-1": 0, "claude-2": 1}]
