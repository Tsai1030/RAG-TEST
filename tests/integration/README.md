# tests/integration

使用真實套件的整合測試，不需要下載模型。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_chromadb.py` | 檔案 | ChromaDB 寫入 3 筆、依 `person_id` 過濾查詢、讀回筆數 | 手動維護 | 是 |
| `test_cli.py` | 檔案 | `memrag models list` 與 `memrag embed --fake --explain` 能執行；未知模型時回報出錯的指令 | 手動維護 | 是 |
| `test_eval_cli.py` | 檔案 | 在暫存資料夾複製真實標準答案，`memrag eval run --fake` 產生完整的結果資料夾；只用開發集；只搜本人的結果不跨人 | 手動維護 | 是 |
| `test_gold_dataset.py` | 檔案 | `data/00_gold/` 的標準答案通過檢查、規模符合計畫、切分與抽樣已寫入 | 手動維護 | 是 |
