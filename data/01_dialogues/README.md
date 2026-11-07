# data/01_dialogues

5.1.4 的模擬對話：陪伴機器人和 20 位虛構使用者在 2026 年 10 月的每日對話。全部是虛構資料。

依計畫「事實先於對話」：每場對話依 `00_gold/gold_facts.jsonl` 的 `mention_date` 寫成，一位使用者一天一場，包含當天提到的 1–2 筆事實和閒聊。閒聊不加入新的個人資訊，以免產生沒有標註的記憶。部分日期用「昨日」「来週の日曜日」等相對說法，用來測試總結時的日期換算。

**撰寫者：** Claude（2026-11-07 使用者決定）。Nano 上的 Qwen3-1.7B、Qwen3.5-2B 試寫時日文不自然、會漏掉事實，不採用。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `dialogues.jsonl` | 檔案 | 一行一場對話（`DialogueLog`）：`log_id`（`d_<使用者>_<月日>`）、`person_id`、`date`、`turns`、`fact_turns` | Claude 撰寫，用 `memrag dialogues validate` 檢查 | 是 |

## 欄位

| 欄位 | 說明 |
|---|---|
| `turns` | `turn_id`（從 1 開始連號）、`speaker`（`user` 或 `robot`）、`text` |
| `fact_turns` | 每筆 Gold Fact 出現在哪幾句；**只供評估**，送給 LLM 的對話不含這個欄位。總結出的記憶依引用的句子對應到 Gold Fact |
