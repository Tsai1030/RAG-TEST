# interfaces/cli/commands

每個檔案是一組子指令，提供 `register()` 註冊參數。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `env.py` | 檔案 | `memrag env check`：環境檢查；在 Nano 上沒有 GPU 就算失敗 | 手動維護 | 是 |
| `models.py` | 檔案 | `memrag models list`、`download`、`probe`：列出、下載與量測模型 | 手動維護 | 是 |
| `embed.py` | 檔案 | `memrag embed`：把一句文字轉成向量；`--explain` 顯示實際送進模型的文字 | 手動維護 | 是 |
| `dataset.py` | 檔案 | `memrag dataset validate`、`show-query`、`split`、`sample`、`agreement`：檢查與查看標準答案 | 手動維護 | 是 |
| `corpus.py` | 檔案 | `memrag dialogues validate`、`memrag summarize`、`memrag corpus build`：檢查對話、總結一場對話（`--show-retries` 看每次嘗試）、總結全部對話並凍結語料 | 手動維護 | 是 |
| `profile.py` | 檔案 | `memrag profile retrieval`（實驗 A：問題到 Top-k 的各段耗時、記憶體、Recall，裝置 × 向量庫 × Top-k）、`profile scale`（實驗 B：記憶庫擴充到各種筆數時各過濾條件的成本） | 手動維護 | 是 |
| `store.py` | 檔案 | `memrag store build`（把記憶庫寫進 `results/vector_db/`）、`store check`（向量庫存檔、重新開啟後對精確搜尋，`--store numpy|chroma`）、`memrag search`（依使用者、類別、日期過濾搜尋記憶） | 手動維護 | 是 |
| `ask.py` | 檔案 | `memrag ask`（完整問答，`--trace` 列出檢索結果、閾值、Context、耗時與設定；`--threshold-mode per_memory|gate` 選閾值做法）、`memrag ingest`（總結對話並寫入向量庫，產生結果資料夾） | 手動維護 | 是 |
| `ui.py` | 檔案 | `memrag ui`：啟動手動測試 UI（`--fake` 不需要模型）；沒安裝 Gradio 時提示 `uv sync --extra ui` | 手動維護 | 是 |
| `evaluate.py` | 檔案 | `memrag eval run`（檢索準確度，`--corpus gold` 或 `memories`）、`eval rescore`（沿用向量重算）、`eval calibrate`／`eval thresholds`（閾值與小候選集）、`eval answers`（每題完整問答，三種做法：逐筆閾值、閾值當閘門、不用閾值；檢查拒答、引用與新舊記憶），以及 `memrag select`（依選型規則排名） | 手動維護 | 是 |
