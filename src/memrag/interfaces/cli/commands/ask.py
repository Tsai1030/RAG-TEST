"""memrag ask | ingest"""

from __future__ import annotations

import memrag

REASONS = {
    "answered": "LLM 依記憶回答",
    "llm_refused": "LLM 判斷記憶沒有回答到問題（第二層）",
    "below_threshold": "沒有記憶通過閾值，未呼叫 LLM（第一層）",
    "no_memory": "過濾範圍內沒有記憶，未呼叫 LLM",
    "invalid_output": "LLM 輸出不符合格式",
}


def register(sub) -> None:
    a = sub.add_parser("ask", help="完整問答：檢索 → 閾值 → Context → LLM 回答與引用（Nano 上要先啟動 LLM）")
    a.add_argument("text")
    a.add_argument("--model", default="ruri-v3-130m")
    a.add_argument("--person", help="使用者 ID，例如 p01；實際系統一定會指定")
    a.add_argument("--from", dest="date_from")
    a.add_argument("--to", dest="date_to")
    a.add_argument("--today", help="今天的日期（YYYY-MM-DD），預設是系統日期；評估資料的問題是 2026-11-01")
    a.add_argument("--no-threshold", action="store_true", help="不套用閾值，只靠 LLM 判斷")
    a.add_argument("--threshold-mode", choices=["per_memory", "gate"],
                   help="per_memory：只送通過閾值的記憶；gate：第一名通過就送全部 Top-k。預設看 pipeline.yaml")
    a.add_argument("--trace", action="store_true", help="列出檢索結果、Context、各階段耗時與設定")
    a.add_argument("--device", default="auto")
    a.add_argument("--fake", action="store_true", help="假 Embedding 與假 LLM，記憶庫放在記憶體裡")
    a.add_argument("--json", action="store_true")
    a.set_defaults(handler=run_ask)

    i = sub.add_parser("ingest", help="總結對話並把記憶寫進向量庫（Nano 上要先啟動 LLM）")
    i.add_argument("log_ids", nargs="*", help="data/01_dialogues 的對話 ID；不給就是全部")
    i.add_argument("--file", help="改用一份對話 JSON 檔")
    i.add_argument("--model", default="ruri-v3-130m")
    i.add_argument("--device", default="auto")
    i.add_argument("--fake", action="store_true")
    i.add_argument("--note", default="")
    i.add_argument("--json", action="store_true")
    i.set_defaults(handler=run_ingest)


def run_ask(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.ask_question(args.text, args.model, args.person, args.date_from, args.date_to, args.today,
                              not args.no_threshold, device=args.device, fake=args.fake,
                              threshold_mode=args.threshold_mode)
    if args.json:
        print_json(out)
        return 0
    print(out["text"])
    if out["citations"]:
        print(f"引用：{', '.join(out['citations'])}")
    if out["invalid_citations"]:
        print(f"無效引用（不在 Context 裡）：{', '.join(out['invalid_citations'])}")
    if not args.trace:
        return 0
    modes = {"per_memory": "逐筆", "gate": "閘門"}
    th = "未校準，不套用" if out["uncalibrated"] else f"{out['threshold']:.4f}（{modes[out['threshold_mode']]}）"
    print(f"\n判斷：{REASONS[out['reason']]}")
    print(f"過濾：{out['filter'] or '無'}　閾值：{th}")
    print("檢索結果：")
    for h in out["retrieved"]:
        mark = "通過" if h["passed"] else "擋下"
        print(f"  {h['score']:.3f} {mark} [{h['memory_id']}] {h['conversation_date']} {h['category']:<12} {h['content']}")
    if out["context"]:
        print("Context：")
        for line in out["context"]:
            print(f"  {line}")
    dropped = {k: v for k, v in out["dropped"].items() if v}
    if dropped:
        print(f"Context 捨棄：{dropped}")
    print("耗時（ms）：" + "、".join(f"{k} {v:.1f}" for k, v in out["timings_ms"].items())
          + ("（LLM 快取）" if out["cached"] else ""))
    m = out["manifest"]
    print(f"設定：{m['embedding_model']['name']}@{m['embedding_model']['revision'][:8]}、"
          f"answer {m['prompts']['answer']['version']} {m['prompts']['answer']['sha256'][:12]}、"
          f"LLM {m['llm']['model']}")
    return 0


def run_ingest(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.ingest_dialogues(args.model, args.log_ids, args.file, device=args.device, fake=args.fake,
                                  note=args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    for r in out["reports"]:
        status = f"{r['accepted']} 筆記憶" if r["ok"] else "失敗（summarize）：" + "；".join(r["errors"])
        print(f"  {r['log_id']}：{status}，重試 {r['retries']} 次，" + "、".join(
            f"{k} {v:.0f} ms" for k, v in r["timings_ms"].items()))
    print(f"對話 {out['logs']} 場，寫入 {out['accepted']} 筆，失敗 {len(out['failed'])} 場；向量庫共 {out['store_count']} 筆")
    return 0 if not out["failed"] else 1
