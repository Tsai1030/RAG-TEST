# tests/integration

使用真實套件的整合測試，不需要下載模型。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_chromadb.py` | 檔案 | ChromaDB 寫入 3 筆、依 `person_id` 過濾查詢、讀回筆數 | 手動維護 | 是 |
| `test_cli.py` | 檔案 | `memrag models list` 與 `memrag embed --fake --explain` 能執行；未知模型時回報出錯的指令 | 手動維護 | 是 |
