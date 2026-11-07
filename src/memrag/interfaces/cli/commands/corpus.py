"""memrag dialogues validate | summarize | corpus build"""

from __future__ import annotations

import memrag


def register(sub) -> None:
    d = sub.add_parser("dialogues", help="模擬對話").add_subparsers(dest="action", required=True)
    v = d.add_parser("validate", help="檢查每筆 Gold Fact 都在正確的人、正確的日期出現一次")
    v.add_argument("--json", action="store_true")
    v.set_defaults(handler=run_validate)

    s = sub.add_parser("summarize", help="把一場對話總結成記憶 JSON")
    s.add_argument("log_id")
    s.add_argument("--fake", action="store_true", help="不呼叫 LLM，把每句使用者發言當成一筆記憶")
    s.add_argument("--no-cache", action="store_true", help="不讀寫快取，一定呼叫 LLM")
    s.add_argument("--show-retries", action="store_true", help="列出每次嘗試的原始輸出與驗證錯誤")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_summarize)

    c = sub.add_parser("corpus", help="記憶語料").add_subparsers(dest="action", required=True)
    b = c.add_parser("build", help="總結全部對話；完整的真實執行會凍結到 data/02_memories/")
    b.add_argument("--fake", action="store_true")
    b.add_argument("--limit", type=int, help="只做前 N 場（試跑，不凍結）")
    b.add_argument("--note", default="")
    b.add_argument("--json", action="store_true")
    b.set_defaults(handler=run_build)


def run_validate(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.validate_dialogues()
    if args.json:
        print_json(out)
    else:
        for i in out["issues"]:
            print(f"[{i['level'].upper():7}] {i['where']}: {i['message']}")
        print(f"對話 {out['logs']} 場、{out['turns']} 句")
        print("結果：" + ("通過" if out["ok"] else "有錯誤"))
    return 0 if out["ok"] else 1


def run_summarize(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.summarize_dialogue(args.log_id, fake=args.fake, use_cache=not args.no_cache)
    if args.json:
        print_json(out)
        return 0 if out["ok"] else 1
    if args.show_retries:
        for n, a in enumerate(out["attempts"], start=1):
            print(f"--- 第 {n} 次{'（快取）' if a['cached'] else ''}：{'通過' if not a['errors'] else '未通過'}")
            print(a["raw"])
            for e in a["errors"]:
                print(f"  ! {e}")
    print(f"{out['log_id']}：{'成功' if out['ok'] else '失敗'}，嘗試 {len(out['attempts'])} 次，記憶 {len(out['memories'])} 筆")
    for m in out["memories"]:
        gold = ",".join(m["gold_fact_ids"]) or "-"
        print(f"  {m['category']:<12} {m['event_date'] or '-':<10} {m['content']}  [turns {m['source_turn_ids']} → {gold}]")
    return 0 if out["ok"] else 1


def run_build(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.build_memory_corpus(fake=args.fake, limit=args.limit, note=args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    for k in ("logs", "success_rate", "first_pass_rate", "retries", "memories", "memories_without_gold_fact",
              "gold_fact_coverage"):
        print(f"{k:<28} {out[k]}")
    print(f"{'failed_logs':<28} {out['failed_logs'] or '無'}")
    print(f"{'uncovered_facts':<28} {len(out['uncovered_facts'])} 筆（--json 可看清單）")
    print("已凍結：" + (out["frozen_sha256"] or "否（試跑或 --fake）"))
    return 0
