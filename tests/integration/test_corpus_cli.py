"""summarize / corpus build / eval --corpus memories run end to end with the fake LLM and fake embeddings."""

import json
import shutil
from pathlib import Path

from memrag.interfaces.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
LOGS = [
    {"log_id": "d_p01_1001", "person_id": "p01", "date": "2026-10-01",
     "turns": [{"turn_id": 1, "speaker": "robot", "text": "佐藤さん、おはようございます。"},
               {"turn_id": 2, "speaker": "user", "text": "主人が亡くなって、もう三年になりますの。"},
               {"turn_id": 3, "speaker": "user", "text": "美穂は世田谷に住んでいて、週に一度電話をくれるのよ。"}],
     "fact_turns": {"f_p01_01": [2], "f_p01_02": [3]}},
    {"log_id": "d_p01_1006", "person_id": "p01", "date": "2026-10-06",
     "turns": [{"turn_id": 1, "speaker": "user", "text": "血圧のお薬、毎朝一錠飲むのを忘れそうで心配なのよ。"}],
     "fact_turns": {"f_p01_06": [1]}},
]


def _project(tmp_path, monkeypatch):
    for folder in ("configs", "prompts"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    shutil.copytree(ROOT / "data" / "00_gold", tmp_path / "data" / "00_gold")
    path = tmp_path / "data" / "01_dialogues" / "dialogues.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(lg, ensure_ascii=False) + "\n" for lg in LOGS), encoding="utf-8")
    monkeypatch.setenv("MEMRAG_ROOT", str(tmp_path))


def test_summarize_and_build_with_fake(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    assert main(["summarize", "d_p01_1001", "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [m["gold_fact_ids"] for m in out["memories"]] == [["f_p01_01"], ["f_p01_02"]]

    assert main(["corpus", "build", "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["success_rate"] == 1.0 and out["gold_fact_coverage"] == 1.0 and out["frozen_sha256"] is None
    run_dir = Path(out["run_dir"])
    assert {p.name for p in run_dir.iterdir()} == {"manifest.json", "attempts.jsonl", "memories.jsonl"}
    assert not (tmp_path / "data" / "02_memories").exists()  # fake runs never freeze

    # Use the fake corpus as if it were frozen, then evaluate on it
    frozen = tmp_path / "data" / "02_memories"
    frozen.mkdir()
    shutil.copy(run_dir / "memories.jsonl", frozen / "memories.jsonl")
    assert main(["eval", "run", "--models", "ruri-v3-30m", "--corpus", "memories", "--split", "all",
                 "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    manifest = json.loads((Path(out["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["docs"] == 3 and manifest["corpus_sha256"]
    # Only queries about these two facts stay answerable; the rest are reported, not scored
    assert out["models"]["ruri-v3-30m"]["person"]["queries"] >= 1
    assert len(manifest["queries_unanswerable_in_corpus"]) > 100

    # Rescoring reuses the saved vectors; a pooled 0 removes the inherited grade
    eval_dir = out["run_dir"]
    memories = [json.loads(line) for line in (frozen / "memories.jsonl").read_text(encoding="utf-8").splitlines()]
    target = next(m for m in memories if m["gold_fact_ids"] == ["f_p01_06"])
    (frozen / "pool_labels.jsonl").write_text(json.dumps(
        {"query_id": "q_p01_01", "memory_id": target["memory_id"], "grade": 0, "annotator": "t"}) + "\n",
        encoding="utf-8")
    assert main(["eval", "rescore", eval_dir, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    manifest = json.loads((Path(out["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_run"] == Path(eval_dir).name and manifest["pool_labels"] == 1
    assert "q_p01_01" in manifest["queries_unanswerable_in_corpus"]

    # A changed corpus is refused
    (frozen / "memories.jsonl").write_text((frozen / "memories.jsonl").read_text(encoding="utf-8")[:-1] + " \n",
                                           encoding="utf-8")
    assert main(["eval", "rescore", eval_dir]) == 2


def test_dialogues_validate_reports_untold_facts(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    assert main(["dialogues", "validate"]) == 1
    assert "沒有出現在任何對話" in capsys.readouterr().out
