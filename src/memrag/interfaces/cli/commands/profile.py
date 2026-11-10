"""memrag profile retrieval | scale | pipeline (plan 5.1.7)"""

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

    pp = p.add_parser("pipeline", help="完整流程（需要 LLM）：總結寫入、Document Embedding 各 batch、"
                                       "問答各階段與 LLM 的 token 速度、量測開銷；在 nano_llm_session.sh 裡執行")
    pp.add_argument("--model", default="ruri-v3-130m")
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--logs", type=int, default=20, help="抽樣幾場對話做總結寫入")
    pp.add_argument("--ks", default="3,5")
    pp.add_argument("--split", choices=["dev", "test", "all"], default="all")
    pp.add_argument("--batch-sizes", default="1,8,32,64")
    pp.add_argument("--warmup", type=int, default=3, help="每個 Top-k 開頭不計入的題數")
    pp.add_argument("--overhead-rounds", type=int, default=2)
    pp.add_argument("--fake", action="store_true")
    pp.add_argument("--note", default="")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(handler=run_pipeline)


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


def run_pipeline(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.profile_pipeline(args.model, args.device, args.logs, _ints(args.ks),
                                  None if args.split == "all" else args.split, _ints(args.batch_sizes), args.warmup,
                                  args.overhead_rounds, args.fake, args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    ing = out["ingest"]
    st = ing["stages"]
    print(f"  總結寫入：{ing['logs']} 場、成功 {ing['ok']}、重試 {ing['retries']}、記憶 {ing['memories']} 筆；"
          f"總結 p50 {_ms(st['summarize'])} ms、Embedding p50 {_ms(st['embed'])} ms、寫入 p50 {_ms(st['store'])} ms")
    for b in out["embedding_batches"]:
        print(f"  Document Embedding batch {b['batch_size']:>3}：{b['docs_per_s']:>8.1f} 筆/秒、CUDA 峰值 {b['cuda_peak_mib']} MiB")
    for k, a in out["answers"].items():
        llm = a["llm"]
        w = a["with_llm"] or {}
        print(f"  Top-{k}：{a['queries']} 題 {a['reasons']}")
        print(f"      有呼叫 LLM：總計 p50 {_ms(w.get('wall'))} ms、p95 {_ms(w.get('wall'), 'p95')} ms；"
              f"首 token p50 {_ms(llm.get('ttft_ms'))} ms、生成 {llm.get('decode_tok_s', '-')} token/秒、"
              f"prompt p50 {_ms(llm.get('prompt_tokens'))} token")
        print(f"      各階段佔比 {w.get('share')}；加總與總計差距 {a['coverage_gap']}")
    m, o = out["memory"], out["overhead"]
    print(f"  記憶體：系統峰值 {m['sys_peak_mib']} MiB、llama-server RSS 峰值 {m['llm_server_rss_peak_mib']} MiB")
    print(f"  量測開銷：每次分段計時 {o['lap_us']} µs；記憶體取樣執行緒讓檢索 p50 變化 {o['sampler']['p50_relative']}")
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
