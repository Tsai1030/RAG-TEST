"""memrag eval run"""

from __future__ import annotations

import memrag

SCOPE_LABELS = {"all": "不過濾（全部 20 人）", "person": "只搜本人", "person+filter": "本人＋日期／類別過濾"}


def register(sub) -> None:
    p = sub.add_parser("eval", help="檢索準確度評估").add_subparsers(dest="action", required=True)

    r = p.add_parser("run", help="把 Gold Facts 當記憶庫，用精確搜尋算各模型的 nDCG、Recall、MRR")
    r.add_argument("--models", default="all", help="all、candidate、baseline，或逗號分隔的模型名稱")
    r.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="測試集只在最終評估使用")
    r.add_argument("--device", default="auto")
    r.add_argument("--fake", action="store_true", help="用假 Embedding 檢查流程，不載入模型")
    r.add_argument("--note", default="")
    r.add_argument("--json", action="store_true")
    r.set_defaults(handler=run_eval)


def run_eval(args) -> int:
    from memrag.interfaces.cli.main import print_json

    out = memrag.evaluate_retrieval(args.models, None if args.split == "all" else args.split,
                                    device=args.device, fake=args.fake, note=args.note)
    if args.json:
        print_json(out)
        return 0
    print(f"結果資料夾：{out['run_dir']}")
    for scope, label in SCOPE_LABELS.items():
        cmp = out["compare_ndcg@10"][scope]
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
