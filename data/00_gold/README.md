# data/00_gold

5.1.3 的標準答案。情境是陪伴機器人與 20 位虛構使用者在 2026 年 10 月的對話；測試問題都在 2026-11-01 提出。規則見 `docs/annotation_guide.md`，欄位定義見 `src/memrag/domain/schemas/evaluation.py`。

**事實先於對話：** 5.1.4 會用這裡的人物設定與 Gold Facts 產生對話，再由 LLM 摘要成記憶，所以標準答案不受摘要結果影響。

```powershell
uv run memrag dataset validate             # 檢查格式與一致性，列出統計
uv run memrag dataset show-query q_p01_05  # 查看一題與它的相關事實
uv run memrag dataset agreement            # 兩次標註的一致度
```

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `personas.jsonl` | 檔案 | 20 位使用者的設定：年齡、家庭、說話方式、簡介 | Claude 撰寫 | 是 |
| `gold_facts.jsonl` | 檔案 | 每人 20 筆、共 400 筆事實，寫法同記憶的 `content`；含哪天說的（`mention_date`）與取代關係（`supersedes`） | Claude 撰寫 | 是 |
| `queries.jsonl` | 檔案 | 201 題日文測試問題，含問題類型、過濾條件、開發集或測試集 | Claude 撰寫；`split` 由 `memrag dataset split --write` 產生 | 是 |
| `labels.jsonl` | 檔案 | 第一次標註（`claude-1`）：只列 1 分與 2 分，沒列出的配對是 0 分 | Claude 標註 | 是 |
| `second_pass_sample.json` | 檔案 | 第二次標註的抽樣題目，依問題類型分層抽 25% | `memrag dataset sample --write` | 是 |
| `labels_second.jsonl` | 檔案 | 第二次獨立標註（`claude-2`），只涵蓋抽樣題目 | 另一個 Claude 工作階段在看不到第一次標註的情況下完成 | 是 |

## 設計

| 項目 | 內容 |
|---|---|
| 事實的類別 | 8 類 `category`，每人 20 筆，類別分布見 `memrag dataset validate` |
| 刻意加入的干擾 | 不同人的相似事實（孫來訪、釣魚、看眼科、健康檢查）、事實更新（`supersedes`，共 9 組）、同一人內的相似事實 |
| 問題類型 | `direct` 40、`paraphrase` 40、`notation` 40、`person` 37、`time_category` 24、`no_answer` 20 |
| 開發集與測試集 | 依問題類型分層，開發集 40%（81 題）、測試集 60%（120 題）；測試集只用於最終評估 |
