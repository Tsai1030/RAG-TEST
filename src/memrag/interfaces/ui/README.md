# interfaces/ui

手動測試 UI（計畫 6.3）。用 Gradio，只綁定 `127.0.0.1`，所有操作都透過 `MemRAG` Facade 與 `memrag` 公開 API。

- **安裝：** Gradio 是選用依賴，`uv sync --extra ui`。
- **啟動：** Nano 上 `uv run memrag ui`，再從 Windows 建立通道 `ssh -L 7860:127.0.0.1:7860 robo@172.28.0.147`，用瀏覽器開 `http://127.0.0.1:7860`。Ask 分頁需要 LLM（`scripts/nano_llm_session.sh`）。
- **不用模型試用：** `uv run memrag ui --fake`（假 Embedding、假 LLM，記憶庫放在記憶體裡）。

| 分頁 | 功能 |
|---|---|
| Ask | 完整問答：回答、判斷（回答、LLM 拒答、閾值擋下）、引用、檢索結果與是否通過閾值、Context、各階段耗時 |
| Search | 依 `person_id`、類別、日期、top_k 搜尋，顯示分數、是否通過閾值與 metadata |
| Compare | 同一查詢並排比較多個模型（每個模型都要先 `memrag store build`） |
| Summarize | 選對話 ID 或貼上對話 JSON，查看每次嘗試、驗證錯誤與最終記憶 |
| Embedding | 查看實際送進模型的文字（prefix）、維度、耗時，算兩句的相似度 |
| Feedback | 把問題 Query 存成候選（`results/feedback/query_candidates.jsonl`），審閱後才加入 Query 集 |

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `app.py` | 檔案 | Gradio 介面：六個分頁與每個模型一個 `MemRAG` 的工作階段 | 手動維護 | 是 |
