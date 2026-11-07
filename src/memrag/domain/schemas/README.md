# domain/schemas

資料結構定義。檢索的 Schema 會在 5.1.5 之後加入。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `embedding.py` | 檔案 | `ModelSpec`（`models.yaml` 的一筆）、`EncodedText`（實際送進模型的文字）與 prefix 規則 | 手動維護 | 是 |
| `memory.py` | 檔案 | 記憶 Schema：`MemoryDraft`（LLM 輸出）、`MemoryRecord`（程式補上 ID 與日期）、8 種 `Category`、`content` 規則 | 手動維護 | 是 |
| `dialogue.py` | 檔案 | 對話紀錄：`DialogueLog`（一位使用者一天的對話）、`Turn`；`render()` 是送給 LLM 的格式；`fact_turns` 只供評估，不給 LLM 看 | 手動維護 | 是 |
| `prompt.py` | 檔案 | `PromptTemplate`：代入變數（缺少或多出都報錯）與內容 hash | 手動維護 | 是 |
| `evaluation.py` | 檔案 | 評估資料：`Persona`、`GoldFact`、`TestQuery`（含過濾條件與切分）、`QueryLabel`（1 或 2 分，沒列出的是 0 分） | 手動維護 | 是 |
