"""`memrag` command entry point. Every command supports --json."""

from __future__ import annotations

import argparse
import json
import sys

from memrag import MemRAGError
from memrag.interfaces.cli.commands import ask, corpus, dataset, embed, env, evaluate, models, store, ui

COMMANDS = [env, models, embed, dataset, corpus, store, ask, evaluate, ui]


def print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memrag", description="日文對話記憶 RAG：Embedding 模型評估")
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except MemRAGError as e:
        print(f"錯誤（{args.command}）：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
