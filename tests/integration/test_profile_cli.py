"""Plan 5.1.7: `memrag profile retrieval | scale` end to end with fake embeddings (child processes included)."""

import json
import shutil
from pathlib import Path

from memrag.interfaces.cli.main import main
from tests.integration.test_corpus_cli import _project


def _frozen_corpus(tmp_path, capsys):
    assert main(["corpus", "build", "--fake", "--json"]) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    (tmp_path / "data" / "02_memories").mkdir()
    shutil.copy(run_dir / "memories.jsonl", tmp_path / "data" / "02_memories" / "memories.jsonl")


def test_profile_retrieval_and_scale_cli(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _frozen_corpus(tmp_path, capsys)

    assert main(["profile", "retrieval", "--fake", "--devices", "cpu", "--repeats", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    rows = {r["store"]: r for r in out["rows"]}
    assert set(rows) == {"chroma", "numpy"} and all(r["ok"] for r in rows.values())
    for r in rows.values():
        assert set(r["latency"]) == {"3", "5"} and r["latency"]["5"]["total"]["n"] > 0
        assert r["by_k"]["5"]["overlap_with_exact"] == 1.0  # tiny corpus: HNSW finds the exact top-k
        assert r["by_k"]["5"]["context_memories_mean"] >= r["by_k"]["3"]["context_memories_mean"]
    assert rows["chroma"]["index_disk_mib"] is not None and rows["numpy"]["index_nbytes_mib"] is not None
    run = Path(out["run_dir"])
    assert {p.name for p in run.iterdir()} == {"manifest.json", "metrics.json", "raw.jsonl"}  # work files removed

    assert main(["profile", "scale", "--fake", "--sizes", "200", "--repeats", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [(r["size"], r["store"], r["ok"]) for r in out["rows"]] == [(200, "chroma", True), (200, "numpy", True)]
    cells = out["rows"][0]["scenarios"]
    assert cells["none"]["selectivity"]["mean_matched"] == 200
    assert cells["person"]["selectivity"]["mean_matched"] == 20  # 10 users
    # A narrow filter may leave nothing for every query; then there is no exact top-k to compare
    assert all((c["overlap_with_exact"]["5"] is None) == (c["selectivity"]["max_matched"] == 0) for c in cells.values())
    assert cells["person"]["overlap_with_exact"]["5"] == 1.0
    assert {p.name for p in Path(out["run_dir"]).iterdir()} == {"manifest.json", "metrics.json", "raw.jsonl"}

    assert main(["profile", "retrieval", "--fake", "--stores", "faiss"]) == 2
