import json
from datetime import date

from memrag.application.summarize import MAX_RETRIES, summarize_log, user_handle, validate_output
from memrag.domain.schemas.dialogue import DialogueLog
from memrag.domain.schemas.prompt import PromptTemplate
from memrag.infrastructure.llm.fake import FakeLLMProvider

LOG = DialogueLog.model_validate({
    "log_id": "d_p01_1001", "person_id": "p01", "date": date(2026, 10, 1),
    "turns": [{"turn_id": 1, "speaker": "robot", "text": "おはようございます。"},
              {"turn_id": 2, "speaker": "user", "text": "来週の土曜日に孫が来るのよ。"},
              {"turn_id": 3, "speaker": "robot", "text": "それは楽しみですね。"},
              {"turn_id": 4, "speaker": "user", "text": "毎朝ラジオ体操をしているの。"}],
    "fact_turns": {"f_p01_05": [2, 3], "f_p01_06": [4]},
})


class Prompts:
    def get(self, name, version):
        if name == "summarize":
            return PromptTemplate(name, version, "draft", ("log_date", "user_name", "dialogue_log", "output_schema"),
                                  "sys", "{{log_date}} {{user_name}}\n{{dialogue_log}}")
        return PromptTemplate(name, version, "draft", ("previous_output", "validation_errors"), None,
                              "retry: {{validation_errors}}")


class DictCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value


def _out(*memories):
    return json.dumps({"memories": list(memories)}, ensure_ascii=False)


PLAN = {"category": "plan", "content": "佐藤さんの孫が10月10日に来る。", "event_date": "2026-10-10",
        "people": ["孫"], "source_turn_ids": [2]}
ROUTINE = {"category": "routine", "content": "佐藤さんは毎朝ラジオ体操をしている。", "event_date": None,
           "people": [], "source_turn_ids": [4]}


def test_first_pass_success_maps_gold_facts():
    llm = FakeLLMProvider([_out(PLAN, ROUTINE)])
    res = summarize_log(LOG, "佐藤さん", llm, Prompts())
    assert res.ok and res.first_pass_ok and len(res.attempts) == 1
    assert [r.gold_fact_ids for r in res.records] == [["f_p01_05"], ["f_p01_06"]]
    assert {r.person_id for r in res.records} == {"p01"}
    assert res.records[0].event_date_ord == 20261010
    # The prompt shows the log but never the fact tags
    sent = llm.requests[0][1]["content"]
    assert "[2] ユーザー：来週の土曜日に孫が来るのよ。" in sent and "f_p01" not in sent


def test_retry_feeds_back_errors_then_succeeds():
    bad_turn = {**PLAN, "source_turn_ids": [9]}
    llm = FakeLLMProvider(["not json", _out(bad_turn), _out(PLAN)])
    res = summarize_log(LOG, "佐藤さん", llm, Prompts())
    assert res.ok and not res.first_pass_ok and len(res.attempts) == 3
    assert "JSON" in res.attempts[0].errors[0]
    assert "[9]" in res.attempts[1].errors[0]
    # The retry message carries the previous output and its errors
    last = llm.requests[2]
    assert last[-2] == {"role": "assistant", "content": _out(bad_turn)}
    assert last[-1]["content"].startswith("retry:") and "[9]" in last[-1]["content"]


def test_gives_up_after_max_retries():
    llm = FakeLLMProvider(["{}"] * 10)
    res = summarize_log(LOG, "佐藤さん", llm, Prompts())
    assert not res.ok and res.records == [] and len(res.attempts) == 1 + MAX_RETRIES


def test_semantic_checks_and_dedupe():
    far = {**PLAN, "event_date": "2028-01-01"}
    _, errors = validate_output(_out(far), LOG)
    assert any("event_date" in e for e in errors)
    chinese = {**PLAN, "content": "佐藤孙子来访"}
    _, errors = validate_output(_out(chinese), LOG)
    assert errors
    drafts, errors = validate_output(_out(PLAN, PLAN, ROUTINE), LOG)
    assert not errors and len(drafts) == 2
    drafts, errors = validate_output(_out(), LOG)
    assert drafts == [] and errors == []


def test_cache_reuses_valid_output_only():
    cache = DictCache()
    summarize_log(LOG, "佐藤さん", FakeLLMProvider(["oops", _out(PLAN)]), Prompts(), cache)
    assert len(cache.data) == 1  # the invalid first output is not cached
    llm = FakeLLMProvider(["oops", _out(ROUTINE)])
    res = summarize_log(LOG, "佐藤さん", llm, Prompts(), cache)
    # First attempt is new (not cached), retry hits the cached valid answer for the same conversation
    assert res.ok and res.attempts[-1].cached and len(llm.requests) == 1


def test_fake_echo_mode():
    res = summarize_log(LOG, "佐藤さん", FakeLLMProvider(), Prompts())
    assert res.ok and [r.source_turn_ids for r in res.records] == [[2], [4]]


def test_user_handle():
    assert user_handle(["さくらちゃんは練馬に住む。", "さくらちゃんの友達は…", "運動会があった。"], "伊藤さん") == "さくらちゃん"
    assert user_handle(["佐藤さんは川越市に住む。"], "x") == "佐藤さん"
    assert user_handle(["散歩した。"], "森さん") == "森さん"
