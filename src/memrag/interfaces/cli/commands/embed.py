"""memrag embed "テスト" --model <name> --kind query --explain"""

from __future__ import annotations

import time

from memrag import MemRAG


def register(sub) -> None:
    p = sub.add_parser("embed", help="把一句文字轉成向量，可查看實際送進模型的文字")
    p.add_argument("text")
    p.add_argument("--model", required=True)
    p.add_argument("--kind", choices=["query", "document"], default="query")
    p.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    p.add_argument("--explain", action="store_true", help="顯示 prefix 與實際送進模型的文字")
    p.add_argument("--fake", action="store_true", help="用 Fake 實作，不載入模型")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=run)


def run(args) -> int:
    from memrag.interfaces.cli.main import print_json

    t = time.perf_counter()
    rag = MemRAG.from_config(args.model, device=args.device, fake=args.fake)
    load_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    vec = rag.embed([args.text], kind=args.kind)[0]
    embed_ms = (time.perf_counter() - t) * 1000
    enc = rag.explain_embedding(args.text, kind=args.kind)
    out = {
        "model": enc.model,
        "kind": enc.kind,
        "dim": len(vec),
        "norm": round(float(sum(x * x for x in vec)) ** 0.5, 4),
        "head": [round(float(x), 4) for x in vec[:8]],
        "load_ms": round(load_ms, 1),
        "embed_ms": round(embed_ms, 1),
    }
    if args.explain:
        out |= {"prefix": enc.prefix, "encoded_text": enc.encoded_text}
    if args.json:
        print_json(out)
    else:
        for k, v in out.items():
            print(f"{k:<13} {v!r}" if k in ("prefix", "encoded_text") else f"{k:<13} {v}")
    return 0
