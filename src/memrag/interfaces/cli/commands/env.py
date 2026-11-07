"""memrag env check"""

from __future__ import annotations

import memrag


def register(sub) -> None:
    p = sub.add_parser("env", help="環境檢查").add_subparsers(dest="action", required=True)
    check = p.add_parser("check", help="Python、套件、GPU 實際運算、ChromaDB 讀寫、模型快取、記憶體與磁碟")
    gpu = check.add_mutually_exclusive_group()
    gpu.add_argument("--require-gpu", dest="require_gpu", action="store_true", default=None,
                     help="沒有 GPU 就算失敗（Nano 上預設開啟）")
    gpu.add_argument("--no-require-gpu", dest="require_gpu", action="store_false")
    check.add_argument("--json", action="store_true")
    check.set_defaults(handler=run_check)


def run_check(args) -> int:
    from memrag.interfaces.cli.main import print_json

    report = memrag.check_environment(require_gpu=args.require_gpu)
    if args.json:
        print_json(report)
    else:
        for c in report["checks"]:
            mark = "OK  " if c["ok"] else ("WARN" if c["name"] == "models" else "FAIL")
            print(f"[{mark}] {c['name']:<10} {c['seconds']:>6.2f}s  {c['detail']}")
        print("結果：" + ("通過" if report["ok"] else "失敗") + "（models 只是提醒，不影響結果）")
    return 0 if report["ok"] else 1
