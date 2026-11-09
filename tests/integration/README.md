# tests/integration

使用真實套件的整合測試，不需要下載模型。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_chromadb.py` | 檔案 | ChromaDB 寫入 3 筆、依 `person_id` 過濾查詢、讀回筆數 | 手動維護 | 是 |
| `test_cli.py` | 檔案 | `memrag models list` 與 `memrag embed --fake --explain` 能執行；未知模型時回報出錯的指令 | 手動維護 | 是 |
| `test_chroma_store.py` | 檔案 | 真實 ChromaDB：200 組隨機過濾（含日期退回對話日期）與精確搜尋接受的記憶完全相同、`get` 讀回完整記錄、只搜本人時不跨人且前 5 名與精確搜尋相同、冪等寫入、空結果、維度與 collection 名稱衝突的保護 | 手動維護 | 是 |
| `test_corpus_cli.py` | 檔案 | 在暫存資料夾用假 LLM 跑 `memrag summarize`、`corpus build`（假資料不凍結），再用 `eval run --corpus memories` 評估；`dialogues validate` 會列出沒出現的事實 | 手動維護 | 是 |
| `test_eval_cli.py` | 檔案 | 在暫存資料夾複製真實標準答案，`memrag eval run --fake` 產生完整的結果資料夾；只用開發集；只搜本人的結果不跨人 | 手動維護 | 是 |
| `test_gold_dataset.py` | 檔案 | `data/00_gold/` 的標準答案通過檢查、規模符合計畫、切分與抽樣已寫入 | 手動維護 | 是 |
| `test_profile_cli.py` | 檔案 | 5.1.7 檢索與規模測試（假 Embedding，含子行程）：`memrag profile retrieval` 兩種向量庫都有各段耗時、Recall 與 Context 長度，`profile scale` 各過濾條件剩下的筆數與 HNSW 對精確搜尋的 Recall；結果資料夾只留 manifest、metrics、raw，暫存檔刪除；不支援的向量庫回報錯誤 | 手動維護 | 是 |
| `test_pipeline_cli.py` | 檔案 | 5.1.6 端到端（假 LLM、假 Embedding、真實 ChromaDB）：寫入兩場對話、搜尋不跨人、回答引用 Context 內的記憶、重跑結果一致、閾值與範圍外拒答、向量庫保存且重複寫入不增加；`memrag ingest`、`ask`、`eval answers` 的結果資料夾與 manifest | 手動維護 | 是 |
