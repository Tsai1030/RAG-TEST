"""`memrag eval run --fake` runs on a copy of the real gold set and writes a complete run folder."""

import json
import shutil
from pathlib import Path

from memrag.interfaces.cli.main import main

ROOT = Path(__file__).resolve().parents[2]


def test_eval_run_fake(tmp_path, monkeypatch, capsys):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "data" / "00_gold", tmp_path / "data" / "00_gold")
    monkeypatch.setenv("MEMRAG_ROOT", str(tmp_path))
    assert main(["eval", "run", "--models", "ruri-v3-30m,e5-small", "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    run_dir = Path(out["run_dir"])
    assert {p.name for p in run_dir.iterdir()} == {"manifest.json", "metrics.json", "per_query.jsonl", "vectors"}
    assert set(out["models"]) == {"ruri-v3-30m", "e5-small"}
    person = out["models"]["ruri-v3-30m"]["person"]
    assert person["queries"] > 0 and 0.0 <= person["ndcg@10"]["mean"] <= 1.0
    rows = [json.loads(line) for line in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    # person scopes only ever return the asker's own memories
    owner = {f["fact_id"]: f["person_id"] for f in map(json.loads, (ROOT / "data/00_gold/gold_facts.jsonl").read_text(encoding="utf-8").splitlines())}
    queries = {q["query_id"]: q for q in map(json.loads, (ROOT / "data/00_gold/queries.jsonl").read_text(encoding="utf-8").splitlines())}
    for r in rows:
        assert queries[r["query_id"]]["split"] == "dev"
        if r["scope"] != "all":
            assert all(owner[t[0]] == queries[r["query_id"]]["person_id"] for t in r["top"])
