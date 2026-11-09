"""Manual-test UI (plan 6.3): Gradio, bound to 127.0.0.1 only. Every action goes through the Facade."""

from __future__ import annotations

import time
from datetime import date

import memrag
from memrag import MemoryFilter, MemRAG

CATEGORIES = ["", "event", "plan", "preference", "relationship", "health", "emotion", "routine", "profile"]
HIT_COLUMNS = ["分數", "閾值", "memory_id", "記錄日", "類別", "日期", "內容"]


class Session:
    """One MemRAG per model, opened on first use and kept for the session."""

    def __init__(self, fake: bool, device: str):
        self.fake, self.device = fake, device
        self._rags: dict[str, MemRAG] = {}
        self._indexed: set[str] = set()

    def rag(self, model: str, indexed: bool = False) -> MemRAG:
        if model not in self._rags:
            self._rags[model] = MemRAG.from_config(model, device=self.device, fake=self.fake)
        rag = self._rags[model]
        if indexed and model not in self._indexed:
            if self.fake:
                rag.index(memrag.load_memories())
            elif rag.store.count() == 0:
                raise memrag.ConfigError(f"{model} 還沒有索引，請先執行 `memrag store build --model {model}`")
            self._indexed.add(model)
        return rag


def _filter(person: str, category: str = "", date_from: str = "", date_to: str = "") -> MemoryFilter:
    return MemoryFilter(person_id=person or None, category=category or None, date_from=date_from or None,
                        date_to=date_to or None)


def _hit_rows(hits, threshold: float | None) -> list[list]:
    return [[round(s, 4), "—" if threshold is None else ("通過" if s >= threshold else "擋下"), m.memory_id,
             str(m.conversation_date), m.category.value, str(m.event_date or ""), m.content] for m, s in hits]


def build(session: Session, default_model: str = "ruri-v3-130m"):
    import gradio as gr

    models = [m["name"] for m in memrag.list_models()]
    log_ids = [lg.log_id for lg in memrag.load_dialogues()]

    def embed(text, model, kind, other):
        rag = session.rag(model)
        start = time.perf_counter()
        vec = rag.embed([text], kind=kind)[0]
        ms = (time.perf_counter() - start) * 1000
        info = rag.explain_embedding(text, kind)
        out = {"encoded_text": info.encoded_text, "dim": len(vec), "ms": round(ms, 1)}
        if other.strip():
            out["cosine_with_second_text"] = round(float(sum(a * b for a, b in zip(vec, rag.embed([other], kind)[0]))), 4)
        return out

    def summarize(log_id, pasted):
        return memrag.summarize_dialogue(log_id, fake=session.fake, log_json=pasted.strip() or None)

    def search(text, model, person, category, date_from, date_to, k):
        rag = session.rag(model, indexed=True)
        hits = rag.search(text, _filter(person, category, date_from, date_to), int(k))
        th = rag.threshold
        note = f"閾值：{'未校準' if th is None else th}；送進模型的文字：{rag.explain_embedding(text, 'query').encoded_text}"
        return note, _hit_rows(hits, th)

    def compare(text, person, chosen, k):
        blocks = []
        for model in chosen:
            rag = session.rag(model, indexed=True)
            rows = _hit_rows(rag.search(text, _filter(person), int(k)), rag.threshold)
            blocks.append(f"### {model}\n" + "\n".join(f"- {r[0]:.3f} {r[1]} {r[6]}" for r in rows))
        return "\n\n".join(blocks) or "請選擇模型"

    def ask(text, model, person, date_from, date_to, today, use_threshold):
        rag = session.rag(model, indexed=True)
        answer = rag.ask(text, _filter(person, "", date_from, date_to),
                         date.fromisoformat(today) if today else None, use_threshold)
        records = {r.memory_id: r for r in rag.store.get([h["memory_id"] for h in answer.retrieved])}
        rows = _hit_rows([(records[h["memory_id"]], h["score"]) for h in answer.retrieved], answer.threshold)
        citations = "、".join(answer.citations) or "無"
        head = f"**{answer.text}**\n\n判斷：`{answer.reason}`　引用：{citations}"
        if answer.invalid_citations:
            head += f"　無效引用：{'、'.join(answer.invalid_citations)}"
        return head, rows, "\n".join(answer.context), {"timings_ms": answer.timings_ms, "cached": answer.cached,
                                                       "filter": answer.filter.model_dump(mode="json", exclude_defaults=True),
                                                       "threshold": answer.threshold, "dropped": answer.dropped}

    def feedback(text, person, problem, model, expected):
        return memrag.save_feedback(text, person, problem, model, expected)

    with gr.Blocks(title="memrag 手動測試") as demo:
        gr.Markdown("# memrag 手動測試" + ("（假模型）" if session.fake else ""))
        with gr.Tab("Ask"):
            with gr.Row():
                q = gr.Textbox(label="問題", value="とんこつラーメン、今もよく食べてるんだっけ？", scale=3)
                m = gr.Dropdown(models, value=default_model, label="模型")
            with gr.Row():
                p = gr.Textbox(label="person_id", value="p06")
                f, t = gr.Textbox(label="日期起（選填）"), gr.Textbox(label="日期迄（選填）")
                today = gr.Textbox(label="今天（評估資料是 2026-11-01）", value="2026-11-01")
                th = gr.Checkbox(label="套用閾值", value=True)
            btn = gr.Button("回答", variant="primary")
            head, hits, ctx, trace = gr.Markdown(), gr.Dataframe(headers=HIT_COLUMNS), gr.Textbox(label="Context"), gr.JSON(label="Trace")
            btn.click(ask, [q, m, p, f, t, today, th], [head, hits, ctx, trace])
        with gr.Tab("Search"):
            with gr.Row():
                sq = gr.Textbox(label="查詢", scale=3)
                sm = gr.Dropdown(models, value=default_model, label="模型")
            with gr.Row():
                sp, sc = gr.Textbox(label="person_id"), gr.Dropdown(CATEGORIES, value="", label="類別")
                sf, st = gr.Textbox(label="日期起"), gr.Textbox(label="日期迄")
                sk = gr.Number(label="top_k", value=5, precision=0)
            sbtn = gr.Button("搜尋", variant="primary")
            snote, shits = gr.Markdown(), gr.Dataframe(headers=HIT_COLUMNS)
            sbtn.click(search, [sq, sm, sp, sc, sf, st, sk], [snote, shits])
        with gr.Tab("Compare"):
            with gr.Row():
                cq, cp = gr.Textbox(label="查詢", scale=3), gr.Textbox(label="person_id")
                ck = gr.Number(label="top_k", value=5, precision=0)
            cm = gr.CheckboxGroup(models, value=[default_model], label="模型（每個都要先 store build）")
            cbtn = gr.Button("比較", variant="primary")
            cout = gr.Markdown()
            cbtn.click(compare, [cq, cp, cm, ck], cout)
        with gr.Tab("Summarize"):
            with gr.Row():
                lid = gr.Dropdown(log_ids, value=log_ids[0] if log_ids else None, label="對話 ID")
                pasted = gr.Textbox(label="或貼上對話 JSON（優先使用）", lines=4, scale=3)
            ubtn = gr.Button("總結", variant="primary")
            uout = gr.JSON(label="每次嘗試、驗證錯誤與最終記憶")
            ubtn.click(summarize, [lid, pasted], uout)
        with gr.Tab("Embedding"):
            with gr.Row():
                et = gr.Textbox(label="文字", value="好きな食べ物", scale=3)
                em = gr.Dropdown(models, value=default_model, label="模型")
                ek = gr.Radio(["query", "document"], value="query", label="kind")
            e2 = gr.Textbox(label="第二句（選填，算相似度）")
            ebtn = gr.Button("向量化", variant="primary")
            eout = gr.JSON()
            ebtn.click(embed, [et, em, ek, e2], eout)
        with gr.Tab("Feedback"):
            gr.Markdown("把手動發現的問題 Query 存成候選（`results/feedback/query_candidates.jsonl`），**審閱後**才可加入 Query 集。")
            with gr.Row():
                ft, fp = gr.Textbox(label="Query", scale=3), gr.Textbox(label="person_id")
            fpr = gr.Textbox(label="問題描述（例如：找不到正確記憶、拒答錯誤）")
            fm, fe = gr.Dropdown(models, value=default_model, label="模型"), gr.Textbox(label="預期結果（選填）")
            fbtn = gr.Button("存成候選")
            fout = gr.JSON()
            fbtn.click(feedback, [ft, fp, fpr, fm, fe], fout)
    return demo


def launch(port: int = 7860, fake: bool = False, device: str = "auto") -> None:
    build(Session(fake, device)).launch(server_name="127.0.0.1", server_port=port, show_api=False)
