# infrastructure

第 3 層。實作 `domain` 的 Port，並處理模型、向量庫、檔案等外部系統。只依賴 `domain`。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `environment.py` | 檔案 | `memrag env check` 的各項檢查：Python、套件、GPU 實際運算、ChromaDB 讀寫、模型快取、記憶體與磁碟 | 手動維護 | 是 |
| `embedding/` | 資料夾 | Embedding 模型相關實作，說明見 [`embedding/README.md`](embedding/README.md) | — | 是 |
| `storage/` | 資料夾 | 執行結果的儲存，說明見 [`storage/README.md`](storage/README.md) | — | 是 |
