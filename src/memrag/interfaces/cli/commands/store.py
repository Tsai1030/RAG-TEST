"""memrag store build | check, memrag search"""

from __future__ import annotations

import memrag


def register(sub) -> None:
    p = sub.add_parser("store", help="向量庫（ChromaDB）").add_subparsers(dest="action", required=True)
    b = p.add_parser("build", help="用指定模型把凍結的記憶庫寫進 results/vector_db/（重跑不會重複）")
    b.add_argument("--model", required=True)
    b.add_argument("--device", default="auto")
    b.add_argument("--fake", action="store_true")
    b.add_argument("--json", action="store_true")
    b.set_defaults(handler=run_build)

    c = p.add_parser("check", help="用某次記憶庫評估存下的向量，比較向量庫（存檔後重新開啟）與精確搜尋：過濾一致、不跨人、Recall")
    c.add_argument("run_dir")
    c.add_argument("--store", choices=["numpy", "chroma"], help="預設用 pipeline.yaml 的 retrieval.store")
    c.add_argument("--note", default="")
    c.add_argument("--json", action="store_true")
    c.set_defaults(handler=run_check)

    s = sub.add_parser("search", help="在向量庫中搜尋記憶")
    s.add_argument("text")
    s.add_argument("--model", required=True)
    s.add_argument("--person")
    s.add_argument("--category")
    s.add_argument("--from", dest="date_from")
    s.add_argument("--to", dest="date_to")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--device", default="auto")
    s.add_argument("--fake", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_search)


def run_build(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.build_index(args.model, device=args.device, fake=args.fake)
    if args.json:
        print_json(out)
    else:
        print(f"{out['model']}：{out['memories']} 筆，向量化 {out['embed_s']} 秒、寫入 {out['upsert_s']} 秒 → {out['path']}")
    return 0


def run_check(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.check_vector_store(args.run_dir, note=args.note, store=args.store)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}（向量庫：{out['store']}）")
    print(f"  {'模型':<20}{'建索引秒':>9}{'磁碟MiB':>9}{'冪等':>5}{'重開筆數':>9}{'過濾不一致':>10}{'跨人':>5}{'R@10 全部':>10}{'R@10 本人':>10}{'R@10 過濾':>10}")
    for model, r in out["models"].items():
        s = r["scopes"]
        print(f"  {model:<20}{r['build_s']:>9.2f}{r['disk_mib']:>9.2f}{'是' if r['idempotent_upsert'] else '否':>5}"
              f"{r['reopened_count']:>9}"
              f"{len(r['filter_mismatches']):>10}{s['person']['cross_user_hits'] + s['person+filter']['cross_user_hits']:>5}"
              f"{s['all']['recall@10']:>10.3f}{s['person']['recall@10']:>10.3f}{s['person+filter']['recall@10']:>10.3f}")
    return 0


def run_search(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.search_memories(args.text, args.model, args.person, args.category, args.date_from, args.date_to,
                                 k=args.k, device=args.device, fake=args.fake)
    if args.json:
        print_json(out)
        return 0
    print(f"送進模型的文字：{out['query']}")
    print(f"過濾：{out['filter'] or '無'}")
    if not out["hits"]:
        print("沒有符合過濾條件的記憶")
    for h in out["hits"]:
        print(f"  {h['score']:.3f}  {h['person_id']} {h['category']:<12} {h['event_date'] or '-':<10} {h['content']}")
    return 0
