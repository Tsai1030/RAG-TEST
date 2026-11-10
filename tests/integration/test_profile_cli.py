"""Plan 5.1.7: `memrag profile retrieval | scale | pipeline` end to end with fake embeddings and LLM
(child processes included)."""

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


def test_profile_pipeline_cli(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _frozen_corpus(tmp_path, capsys)
    monkeypatch.setenv("RAG_LLM_LOAD_S", "12.5")

    assert main(["profile", "pipeline", "--fake", "--logs", "3", "--split", "dev", "--overhead-rounds", "1",
                 "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["load"]["llm_session"]["load_s"] == 12.5 and out["load"]["llm_session"]["ctx"] is None
    ing = out["ingest"]
    assert 0 < ing["logs"] <= 3 and ing["ok"] == ing["llm_calls"] == ing["logs"]  # fewer when the project has fewer logs
    assert out["embedding_batches"] == []  # the fake embedder has no batch size
    for k in ("3", "5"):
        a = out["answers"][k]
        assert a["queries"] == sum(a["reasons"].values()) > 0
        assert a["llm"]["calls"] == a["reasons"].get("answered", 0) + a["reasons"].get("llm_refused", 0)
        # The < 5% acceptance applies to real runs; fake stages take microseconds, so any wrapper cost dominates
        assert 0 <= a["coverage_gap"]["mean"] <= a["coverage_gap"]["max"] < 1
    assert out["memory"]["llm_server_rss_peak_mib"] is None
    assert out["overhead"]["lap_us"] > 0 and out["overhead"]["sampler"]["off"]["n"] > 0
    run = Path(out["run_dir"])
    assert {p.name for p in run.iterdir()} == {"manifest.json", "metrics.json", "ingest.jsonl", "answers.jsonl"}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["logs"]) == ing["logs"] and manifest["response_cache"] is False
