"""memrag models list | download | probe"""

from __future__ import annotations

import memrag

SELECTION_HELP = "all、candidate、baseline，或以逗號分隔的模型名稱"


def register(sub) -> None:
    p = sub.add_parser("models", help="模型 registry、下載與載入量測").add_subparsers(dest="action", required=True)

    ls = p.add_parser("list", help="列出模型、prefix 與是否已下載")
    ls.add_argument("--models", default="all", help=SELECTION_HELP)
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(handler=run_list)

    dl = p.add_parser("download", help="下載固定 revision 的模型到專案的 .hf_cache")
    dl.add_argument("--models", default="all", help=SELECTION_HELP)
    dl.add_argument("--json", action="store_true")
    dl.set_defaults(handler=run_download)

    pr = p.add_parser("probe", help="每個模型在獨立子行程量測載入時間、記憶體與延遲")
    pr.add_argument("--models", default="all", help=SELECTION_HELP)
    pr.add_argument("--devices", default="cuda,cpu", help="以逗號分隔，例如 cuda,cpu")
    pr.add_argument("--note", default="", help="寫入 manifest 的備註，例如常駐服務是否已停止")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(handler=run_probe)


def run_list(args) -> int:
    from memrag.interfaces.cli.main import print_json

    rows = memrag.list_models(args.models)
    if args.json:
        print_json(rows)
        return 0
    for r in rows:
        mark = "已下載" if r["downloaded"] else "未下載"
        print(f"{r['name']:<18} {r['role']:<9} dim={r['dim']:<5} {r['license']:<11} {mark}  {r['hf_id']}@{r['revision'][:8]}")
        print(f"    query={r['query_prefix']!r}  document={r['document_prefix']!r}")
    return 0


def run_download(args) -> int:
    from memrag.interfaces.cli.main import print_json

    rows = memrag.download_models(args.models)
    if args.json:
        print_json(rows)
    else:
        for r in rows:
            print(f"{r['model']:<18} {r['size_mib']:>8.1f} MiB  {r['path']}")
    return 0


def run_probe(args) -> int:
    from memrag.interfaces.cli.main import print_json

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out = memrag.probe_models(args.models, devices, note=args.note)
    if args.json:
        print_json(out)
    else:
        for r in out["results"]:
            if not r["ok"]:
                print(f"{r['model']:<18} {r['device']:<5} 失敗：{r['error']}")
                continue
            print(f"{r['model']:<18} {r['device']:<5} {r['dtype']:<9} load={r['load_s']:>6.2f}s "
                  f"p50={r['latency_ms_p50']:>7.1f}ms p95={r['latency_ms_p95']:>7.1f}ms "
                  f"sys_peak={r['sys_mem_peak_mib']:>7.1f}MiB cuda_peak={r['cuda_peak_mib']}")
        print(f"結果：{out['run_dir']}")
    return 0 if all(r["ok"] for r in out["results"]) else 1
