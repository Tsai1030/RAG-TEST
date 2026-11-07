# domain

第 1 層。定義資料結構與 Port 介面，不得 import `torch`、`chromadb`、`sentence_transformers`、`yaml` 等外部套件，也不得依賴其他層。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `ports.py` | 檔案 | Port 介面：目前有 `EmbeddingProvider`，其他 Port 隨各任務加入 | 手動維護 | 是 |
| `errors.py` | 檔案 | 領域錯誤：`MemRAGError`、`ConfigError`、`ModelNotAvailableError` | 手動維護 | 是 |
| `schemas/` | 資料夾 | 資料結構，說明見 [`schemas/README.md`](schemas/README.md) | — | 是 |
