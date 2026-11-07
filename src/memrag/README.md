# memrag

主套件，依計畫書第 4 節分層：Interface → Facade → Application → Domain，Infrastructure 實作 Domain 的 Port。分層規則由 `tests/architecture/test_layers.py` 檢查。

公開 API 只從 `__init__.py` 匯出，其他模組視為內部實作。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 匯出公開 API：`MemRAG`、`list_models`、`check_environment`、`download_models`、`probe_models` 與錯誤類別 | 手動維護 | 是 |
| `api.py` | 檔案 | Facade：`MemRAG` 類別與模型、環境相關函式；CLI 與 UI 只呼叫這裡 | 手動維護 | 是 |
| `bootstrap.py` | 檔案 | 依設定組裝依賴：找專案根目錄、讀 registry、建立 `EmbeddingProvider` | 手動維護 | 是 |
| `domain/` | 資料夾 | 第 1 層：Schema 與 Port，不依賴外部框架，說明見 [`domain/README.md`](domain/README.md) | — | 是 |
| `application/` | 資料夾 | 第 2 層：Use case，說明見 [`application/README.md`](application/README.md) | — | 是 |
| `infrastructure/` | 資料夾 | 第 3 層：模型、向量庫等外部系統實作，說明見 [`infrastructure/README.md`](infrastructure/README.md) | — | 是 |
| `interfaces/` | 資料夾 | 第 4 層：CLI 等使用者入口，說明見 [`interfaces/README.md`](interfaces/README.md) | — | 是 |
