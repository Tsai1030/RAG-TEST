"""memrag profile retrieval | scale (plan 5.1.7 retrieval and scale test)"""

from __future__ import annotations

import memrag


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def register(sub) -> None:
    p = sub.add_parser("profile", help="效能量測（5.1.7）").add_subparsers(dest="action", required=True)
    r = p.add_parser("retrieval", help="實驗 A：問題到 Top-k 的各段耗時、記憶體、Recall（現有記憶庫，每個組合一個子行程）")
    r.add_argument("--model", default="ruri-v3-130m")
    r.add_argument("--devices", default="cpu,cuda", help="以逗號分隔，例如 cpu,cuda")
    r.add_argument("--stores", default="chroma,numpy", help="chroma（HNSW）、numpy（精確搜尋）")
    r.add_argument("--ks", default="3,5", help="Top-k，以逗號分隔")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--split", choices=["dev", "test", "all"], default="all")
    r.add_argument("--fake", action="store_true")
    r.add_argument("--note", default="")
    r.add_argument("--json", action="store_true")
    r.set_defaults(handler=run_retrieval)

    s = p.add_parser("scale", help="實驗 B：資料擴充到各種筆數時，各種過濾條件的耗時、記憶體與 HNSW Recall")
    s.add_argument("--model", default="ruri-v3-130m")
    s.add_argument("--sizes", default="1000,10000,50000", help="以逗號分隔")
    s.add_argument("--stores", default="chroma,numpy")
    s.add_argument("--ks", default="3,5")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--device", default="cpu", help="計算基礎向量用的裝置（不影響量測）")
    s.add_argument("--fake", action="store_true")
    s.add_argument("--note", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_scale)


def _ms(cell: dict | None, key: str = "p50") -> str:
    return f"{cell[key]:.2f}" if cell and cell.get(key) is not None else "-"


def run_retrieval(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.profile_retrieval(args.model, _names(args.devices), _names(args.stores), _ints(args.ks), args.repeats,
                                   None if args.split == "all" else args.split, args.fake, args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    print(f"  {'裝置':<6}{'向量庫':<8}{'K':>3}{'Embed p50':>11}{'檢索 p50':>10}{'取回 p50':>10}{'總計 p50':>10}"
          f"{'總計 p95':>10}{'QPS':>8}{'Recall':>8}{'最佳在前K':>10}{'對精確':>8}{'峰值MiB':>9}")
    for row in out["rows"]:
        if not row.get("ok"):
            print(f"  {row['device']:<6}{row['store']:<8} 失敗：{row.get('error')}")
            continue
        for k, lat in row["latency"].items():
            q = row["by_k"][k]
            print(f"  {row['device']:<6}{row['store']:<8}{k:>3}{_ms(lat['embed']):>11}{_ms(lat['search']):>10}"
                  f"{_ms(lat['get']):>10}{_ms(lat['total']):>10}{_ms(lat['total'], 'p95'):>10}{lat['qps']:>8.1f}"
                  f"{q['recall'] or 0:>8.3f}{q['best_in_top_k'] or 0:>10.3f}"
                  f"{q['overlap_with_exact'] if q['overlap_with_exact'] is not None else '-':>8}"
                  f"{row['sys_mem_peak_mib']:>9.0f}")
    return 0


def run_scale(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.profile_scale(args.model, _ints(args.sizes), _names(args.stores), _ints(args.ks), args.repeats,
                               args.device, args.fake, args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    k = str(max(_ints(args.ks)))
    for row in out["rows"]:
        if not row.get("ok"):
            print(f"  {row['size']:>6} {row['store']:<7} 失敗：{row.get('error')}")
            continue
        print(f"  {row['size']:>6} 筆 {row['store']:<7} 寫入 {row['upsert_s']:.2f} s、索引記憶體 {row['index_mem_mib']} MiB、"
              f"磁碟 {row['index_disk_mib'] or '-'} MiB、峰值 {row['sys_mem_peak_mib']} MiB")
        for s, cell in row["scenarios"].items():
            lat = cell["latency"][k]
            sel = cell.get("selectivity") or {}
            ov = (cell.get("overlap_with_exact") or {}).get(k)
            print(f"      {s:<24} 剩 {sel.get('mean_matched', '-'):>8} 筆  檢索 p50 {_ms(lat['search']):>8} ms"
                  f"  p95 {_ms(lat['search'], 'p95'):>8} ms  Recall@{k} 對精確 {ov if ov is not None else '-'}")
    return 0
