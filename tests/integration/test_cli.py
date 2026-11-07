"""CLI entry points work end to end with the Fake provider (no model download needed)."""

import json

from memrag.interfaces.cli.main import main


def test_models_list_json(capsys):
    assert main(["models", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in rows} >= {"ruri-v3-30m", "bge-base-zh-v1.5"}


def test_embed_explain_with_fake(capsys):
    assert main(["embed", "テスト", "--model", "ruri-v3-30m", "--kind", "query", "--explain", "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["encoded_text"] == "検索クエリ: テスト"
    assert out["dim"] == 256
    assert abs(out["norm"] - 1.0) < 1e-3


def test_unknown_model_reports_stage(capsys):
    assert main(["embed", "テスト", "--model", "nope", "--fake"]) == 2
    assert "錯誤（embed）" in capsys.readouterr().err
