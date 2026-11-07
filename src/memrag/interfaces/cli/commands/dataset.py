"""memrag dataset validate | show-query | split | agreement"""

from __future__ import annotations

import memrag


def register(sub) -> None:
    p = sub.add_parser("dataset", help="評估資料：人物設定、Gold Facts、測試問題與標註").add_subparsers(dest="action", required=True)

    v = p.add_parser("validate", help="檢查格式、ID、標註與過濾條件是否一致，並列出統計")
    v.add_argument("--json", action="store_true")
    v.set_defaults(handler=run_validate)

    s = p.add_parser("show-query", help="顯示一題測試問題與標為相關的事實")
    s.add_argument("query_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_show)

    sp = p.add_parser("split", help="依問題類型分層切出開發集與測試集")
    sp.add_argument("--write", action="store_true", help="寫回 queries.jsonl；沒加只預覽")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(handler=run_split)

    sm = p.add_parser("sample", help="依問題類型分層抽出 25% 給第二次獨立標註")
    sm.add_argument("--write", action="store_true", help="寫入 second_pass_sample.json；沒加只預覽")
    sm.add_argument("--json", action="store_true")
    sm.set_defaults(handler=run_sample)

    a = p.add_parser("agreement", help="兩次標註的一致度（Cohen's kappa）")
    a.add_argument("--json", action="store_true")
    a.set_defaults(handler=run_agreement)


def run_validate(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.validate_dataset()
    if args.json:
        print_json(out)
    else:
        for i in out["issues"]:
            print(f"[{i['level'].upper():7}] {i['where']}: {i['message']}")
        for k, v in out["stats"].items():
            print(f"{k:<28} {v}")
        print("結果：" + ("通過" if out["ok"] else "有錯誤"))
    return 0 if out["ok"] else 1


def run_show(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.show_query(args.query_id)
    if args.json:
        print_json(out)
        return 0
    q, p = out["query"], out["persona"]
    print(f"{q['query_id']}  {q['query_type']}  {q['split']}  {p['person_id']} {p['name']}（{p['age']}）  asked_on={q['asked_on']}")
    print(f"  問題：{q['text']}")
    flt = {k: v for k, v in q["filters"].items() if v}
    if flt:
        print(f"  過濾：{flt}")
    if q["note"]:
        print(f"  說明：{q['note']}")
    if not out["relevant"]:
        print("  相關事實：無（無答案題）")
    for f in out["relevant"]:
        print(f"  [{f['grade']}] {f['fact_id']} {f['category']:<12} {f['event_date'] or '-':<10} {f['statement']}")
    return 0


def run_split(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.assign_dataset_splits(write=args.write)
    if args.json:
        print_json(out)
    else:
        print(f"dev/test：{out['counts']}，變更 {out['changed']} 題，已寫入：{out['written']}")
    return 0


def run_sample(args) -> int:
    from memrag.interfaces.cli.main import print_json

    sample = memrag.make_second_pass_sample(write=args.write)
    if args.json:
        print_json(sample)
    else:
        print(f"抽出 {len(sample)} 題，已寫入：{args.write}")
    return 0


def run_agreement(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.annotation_agreement()
    if args.json:
        print_json(out)
    else:
        for k, v in out.items():
            if k != "disagreements":
                print(f"{k:<24} {v}")
        print(f"不一致的配對：{len(out['disagreements'])} 組（--json 可看明細）")
    return 0
