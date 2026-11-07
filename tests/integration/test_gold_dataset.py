"""The committed gold set in data/00_gold/ is valid and matches the plan's targets."""

import json
from pathlib import Path

import pytest

import memrag

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def result():
    return memrag.validate_dataset()


def test_gold_set_has_no_errors(result):
    assert result["ok"], [i for i in result["issues"] if i["level"] == "error"]


def test_gold_set_size(result):
    s = result["stats"]
    assert s["personas"] == 20
    assert 150 <= s["queries"] <= 210
    assert s["facts_per_person"]["min"] >= 20


def test_dev_test_split_written(result):
    splits = result["stats"]["splits"]
    assert set(splits) == {"dev", "test"}
    assert 0.35 <= splits["dev"] / result["stats"]["queries"] <= 0.45


def test_second_pass_sample_is_at_least_20_percent(result):
    sample = json.loads((ROOT / "data" / "00_gold" / "second_pass_sample.json").read_text(encoding="utf-8"))
    assert len(sample) >= 0.2 * result["stats"]["queries"]
    assert sample == memrag.make_second_pass_sample()
