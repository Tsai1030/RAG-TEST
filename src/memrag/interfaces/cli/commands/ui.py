"""memrag ui"""

from __future__ import annotations


def register(sub) -> None:
    u = sub.add_parser("ui", help="啟動手動測試 UI（Gradio，只綁定 127.0.0.1；需要 uv sync --extra ui）")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--device", default="auto")
    u.add_argument("--fake", action="store_true", help="假 Embedding 與假 LLM，不需要模型與 LLM 伺服器")
    u.set_defaults(handler=run_ui)


def run_ui(args) -> int:
    try:
        import gradio  # noqa: F401
    except ImportError:
        print("錯誤（ui）：沒有安裝 Gradio，請先執行 uv sync --extra ui")
        return 2
    from memrag.interfaces.ui.app import launch

    launch(args.port, fake=args.fake, device=args.device)
    return 0
