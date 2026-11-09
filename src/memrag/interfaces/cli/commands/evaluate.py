"""memrag eval run | rescore | calibrate | thresholds | answers, memrag select"""

from __future__ import annotations

import memrag

SCOPE_LABELS = {"all": "不過濾（全部 20 人）", "person": "只搜本人", "person+filter": "本人＋日期／類別過濾"}


def register(sub) -> None:
    register_select(sub)
    p = sub.add_parser("eval", help="檢索準確度評估").add_subparsers(dest="action", required=True)

    r = p.add_parser("run", help="用精確搜尋算各模型的 nDCG、Recall、MRR")
    r.add_argument("--models", default="all", help="all、candidate、baseline，或逗號分隔的模型名稱")
    r.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="測試集只在最終評估使用")
    r.add_argument("--corpus", default="gold", choices=["gold", "memories"],
                   help="gold：Gold Facts 原文；memories：LLM 總結出的凍結語料")
    r.add_argument("--device", default="auto")
    r.add_argument("--fake", action="store_true", help="用假 Embedding 檢查流程，不載入模型")
    r.add_argument("--note", default="")
    r.add_argument("--json", action="store_true")
    r.set_defaults(handler=run_eval)

    s = p.add_parser("rescore", help="用目前的標註重算一次已完成的評估（沿用存下的向量，不載入模型）")
    s.add_argument("run_dir", help="results/runs/<run_id> 的路徑")
    s.add_argument("--note", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_rescore)

    c = p.add_parser("calibrate", help="在開發集的小候選集（2–10 筆）上校準各模型的閾值（無答案誤召回率 ≤ 5%）")
    c.add_argument("run_dir", help="開發集、--corpus memories 的 results/runs/<run_id>")
    c.add_argument("--write", action="store_true", help="寫入 configs/thresholds.yaml；沒加只預覽")
    c.add_argument("--note", default="")
    c.add_argument("--json", action="store_true")
    c.set_defaults(handler=run_calibrate)

    t = p.add_parser("thresholds", help="用 configs/thresholds.yaml 的固定閾值評估小候選集（例如測試集）")
    t.add_argument("run_dir")
    t.add_argument("--note", default="")
    t.add_argument("--json", action="store_true")
    t.set_defaults(handler=run_thresholds)

    a = p.add_parser("answers", help="每題都跑完整問答，檢查拒答與引用（計畫 5.1.6；Nano 上要先啟動 LLM）")
    a.add_argument("--model", default="ruri-v3-130m")
    a.add_argument("--split", default="all", choices=["dev", "test", "all"])
    a.add_argument("--limit", type=int, help="只跑前 N 題，用來試跑")
    a.add_argument("--device", default="auto")
    a.add_argument("--fake", action="store_true")
    a.add_argument("--note", default="")
    a.add_argument("--json", action="store_true")
    a.set_defaults(handler=run_answers)


def run_answers(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.evaluate_answers(args.model, None if args.split == "all" else args.split, args.limit,
                                  device=args.device, fake=args.fake, note=args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    print(f"  {'做法':<16}{'有答案→回答':>12}{'引用相關':>9}{'無答案→拒答':>12}{'部分相關→回答':>14}{'無效引用':>9}{'過濾外':>7}{'更新題引用新':>12}")
    for variant, m in out["variants"].items():
        u = m["updates"]
        print(f"  {variant:<16}{_fmt(m['answerable']['answered']):>12}{_fmt(m['answerable']['cited_relevant']):>9}"
              f"{_fmt(m['no_answer']['refused']):>12}{_fmt(m['partial']['answered']):>14}"
              f"{m['citations']['invalid']:>9}{m['citations']['outside_filter']:>7}{u['cited_new']:>8}/{u['answered']}")
    r = out["retrieval_repeat"]
    print(f"重跑檢索：{r['queries']} 題，結果不同 {len(r['different'])} 題")
    return 0


def register_select(sub) -> None:
    s = sub.add_parser("select", help="依 configs/selection.yaml 做選型：硬性門檻與加權排名（計畫 5.1.9）")
    s.add_argument("--retrieval", required=True, help="測試集、記憶庫的 eval run 或 eval rescore 結果資料夾")
    s.add_argument("--thresholds", required=True, help="同一批向量的 eval thresholds 結果資料夾")
    s.add_argument("--probe", required=True, help="models probe 的結果資料夾")
    s.add_argument("--note", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=run_select)


def run_select(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.select_models(args.retrieval, args.thresholds, args.probe, note=args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    print(f"  {'模型':<20}{'排名':>4}{'分數':>8}{'nDCG@10':>9}{'R@5':>7}{'誤擋':>7}{'p95 ms':>8}{'MiB':>7}  門檻")
    for m in sorted(out["models"], key=lambda m: m.get("rank", 99)):
        rank = m.get("rank", "—")
        score = f"{m['score']:.3f}" if "score" in m else "—"
        gates = "；".join(m["gate_failures"]) or ("通過" if m["role"] == "candidate" else "對照組，不排名")
        print(f"  {m['name']:<20}{rank:>4}{score:>8}{m['ndcg@10']:>9.3f}{m['recall@5']:>7.3f}"
              f"{m['small_set_false_block']:>7.3f}{m['query_latency_p95_ms']:>8.1f}{m['memory_peak_mib']:>7.0f}  {gates}")
    stable = all(r["same"] for r in out["sensitivity"])
    print(f"權重 ±0.05 敏感度：{'第一名不變' if stable else '第一名會改變：' + '、'.join(r['metric'] + r['change'] + '→' + r['first'] for r in out['sensitivity'] if not r['same'])}")
    return 0


def run_calibrate(args) -> int:
    out = memrag.calibrate_thresholds(args.run_dir, write=args.write, note=args.note)
    _print_thresholds(out, args.json)
    if not args.json:
        print("已寫入 configs/thresholds.yaml" if out["written"] else "預覽：加 --write 才會寫入 configs/thresholds.yaml")
    return 0


def run_thresholds(args) -> int:
    return _print_thresholds(memrag.evaluate_thresholds(args.run_dir, note=args.note), args.json)


def _fmt(rate: dict | None) -> str:
    return "     —" if rate is None or "rate" not in rate else f"{rate['rate']:6.3f}"


def _print_thresholds(out: dict, as_json: bool) -> int:
    from memrag.interfaces.cli.main import print_json

    if as_json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    print(f"  {'模型':<24}{'閾值':>7}{'無答誤召回':>10}{'正確誤擋':>9}{'誤擋(2筆)':>10}{'誤擋(10筆)':>11}{'錯誤通過':>9}{'Top-1錯':>8}")
    for model, r in out["models"].items():
        a, s2, s10 = r["by_size"]["all"], r["by_size"].get("2"), r["by_size"].get("10")
        print(f"  {model:<24}{r['threshold']:>7.3f}{_fmt(a['false_recall']):>10}{_fmt(a['false_block']):>9}"
              f"{_fmt(s2 and s2['false_block']):>10}{_fmt(s10 and s10['false_block']):>11}"
              f"{a['wrong_pass_rate'] if a['wrong_pass_rate'] is not None else float('nan'):>9.3f}{_fmt(a['top1_wrong']):>8}")
    return 0


def run_eval(args) -> int:
    out = memrag.evaluate_retrieval(args.models, None if args.split == "all" else args.split,
                                    device=args.device, fake=args.fake, note=args.note, corpus=args.corpus)
    return _print(out, args.json)


def run_rescore(args) -> int:
    return _print(memrag.rescore_retrieval(args.run_dir, note=args.note), args.json)


def _print(out: dict, as_json: bool) -> int:
    from memrag.interfaces.cli.main import print_json

    if as_json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    for scope, label in SCOPE_LABELS.items():
        cmp = out["compare_ndcg@10"].get(scope)
        if cmp is None:
            print(f"\n[{label}]  沒有可評估的有答案題目")
            continue
        print(f"\n[{label}]  最佳：{cmp['best']}")
        print(f"  {'模型':<24}{'nDCG@10':>9}{'95% CI':>17}{'R@5':>7}{'R@1':>7}{'MRR':>7}{'錯人':>6}{'無答AUC':>8}  與最佳差距")
        for model, m in out["models"].items():
            s = m[scope]
            ci = s["ndcg@10"]["ci95"]
            d = cmp["vs_best"][model]
            mark = "顯著較差" if d["significant"] else ("—" if model == cmp["best"] else "不顯著")
            print(f"  {model:<24}{s['ndcg@10']['mean']:>9.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]"
                  f"{s['recall@5']['mean']:>7.3f}{s['recall@1']['mean']:>7.3f}{s['mrr@10']['mean']:>7.3f}"
                  f"{s['top1_other_person_rate']:>6.2f}{s.get('no_answer_auc', float('nan')):>8.3f}  {d['diff']:+.3f} {mark}")
    return 0
