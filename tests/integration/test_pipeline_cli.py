"""Plan 5.1.6: the whole pipeline (ingest → search → ask) and its CLI with the fake LLM and fake embeddings."""

import json
import shutil
from datetime import date
from pathlib import Path

from memrag import MemoryFilter, MemRAG
from memrag.interfaces.cli.main import main
from tests.integration.test_corpus_cli import LOGS, _project


def test_ingest_search_ask_with_fakes(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    from memrag.domain.schemas.dialogue import DialogueLog

    rag = MemRAG.from_config("ruri-v3-30m", fake=True, store_dir=tmp_path / "db")
    reports = [rag.ingest_log(DialogueLog.model_validate(lg)) for lg in LOGS]
    assert [(r.ok, r.accepted, r.retries) for r in reports] == [(True, 2, 0), (True, 1, 0)]
    assert set(reports[0].timings_ms) == {"summarize", "embed", "store"}
    flt = MemoryFilter(person_id="p01")
    hits = rag.search("お薬のこと", flt, top_k=5)
    assert len(hits) == 3 and all(m.person_id == "p01" for m, _ in hits)

    # The fake LLM answers with the first context memory and cites it
    answer = rag.ask("お薬のこと、覚えてる？", flt, date(2026, 11, 1), use_threshold=False)
    assert answer.reason == "answered" and answer.uncalibrated
    assert answer.citations == [answer.retrieved[0]["memory_id"]] and answer.invalid_citations == []
    again = rag.ask("お薬のこと、覚えてる？", flt, date(2026, 11, 1), use_threshold=False)
    assert again.retrieved == answer.retrieved and again.citations == answer.citations
    # Fake vectors never pass a real threshold, and nobody else's memories are visible
    assert rag.ask("お薬のこと、覚えてる？", flt).reason == "below_threshold"
    assert rag.ask("お薬", MemoryFilter(person_id="p02")).reason == "no_memory"
    assert rag.pipeline_filter(MemoryFilter(person_id="p01", category="plan")) == MemoryFilter(
        person_id="p01", date_fallback=True)  # 5.1.6: no category filter, date fallback
    rag.close()

    # The store persists; ingesting the same log again adds nothing
    rag = MemRAG.from_config("ruri-v3-30m", fake=True, store_dir=tmp_path / "db")
    assert rag.store.count() == 3
    rag.ingest_log(DialogueLog.model_validate(LOGS[0]))
    assert rag.store.count() == 3
    rag.close()


def test_ask_ingest_and_answer_eval_cli(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    assert main(["ingest", "d_p01_1001", "--fake", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["accepted"] == 2 and out["failed"] == []
    manifest = json.loads((Path(out["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_model"]["revision"] and manifest["prompts"]["summarize"]["sha256"]
    assert main(["ingest", "d_nope", "--fake"]) == 2

    # ask and eval answers read the frozen corpus; use a fake one
    assert main(["corpus", "build", "--fake", "--json"]) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    (tmp_path / "data" / "02_memories").mkdir()
    shutil.copy(run_dir / "memories.jsonl", tmp_path / "data" / "02_memories" / "memories.jsonl")

    assert main(["ask", "血圧のお薬は？", "--person", "p01", "--fake", "--no-threshold", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reason"] == "answered" and out["citations"][0] == out["retrieved"][0]["memory_id"]
    assert out["retrieved"][0]["person_id"] == "p01" and out["uncalibrated"]  # --no-threshold
    assert out["manifest"]["threshold"]["value"] == 0.849  # the configured one for ruri-v3-130m (default)
    assert main(["ask", "血圧のお薬は？", "--person", "p01", "--fake", "--trace"]) == 0
    assert "第一層" in capsys.readouterr().out

    assert main(["eval", "answers", "--fake", "--limit", "6", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["variants"]) == {"threshold+llm", "gate+llm", "llm_only"}
    for m in out["variants"].values():
        assert m["queries"] == 6 and m["citations"]["invalid"] == 0 and m["citations"]["outside_filter"] == 0
    assert out["retrieval_repeat"]["different"] == []
    files = {p.name for p in Path(out["run_dir"]).iterdir()}
    assert files == {"manifest.json", "answers.jsonl", "metrics.json"}
