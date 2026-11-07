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
| `store.py` | 檔案 | `memrag store build`（把記憶庫寫進 `results/vector_db/`）、`store check`（ChromaDB 對精確搜尋）、`memrag search`（依使用者、類別、日期過濾搜尋記憶） | 手動維護 | 是 |
| `evaluate.py` | 檔案 | `memrag eval run`（檢索準確度，`--corpus gold` 或 `memories`）、`eval rescore`（沿用向量重算）、`eval calibrate`／`eval thresholds`（閾值與小候選集），以及 `memrag select`（依選型規則排名） | 手動維護 | 是 |
