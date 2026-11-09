# infrastructure/vectorstore

`VectorStore` 的實作（計畫 5.1.5）。過濾的語意以 `domain/schemas/retrieval.py` 的 `MemoryFilter.matches` 為準：日期條件會排除沒有 `event_date` 的記憶。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `exact_store.py` | 檔案 | 記憶體內的精確 cosine 搜尋，是過濾與 Recall 的對照基準 | 手動維護 | 是 |
| `numpy_store.py` | 檔案 | 產品用的向量庫（5.1.7 起，`pipeline.yaml` 的 `retrieval.store: numpy`）：精確搜尋，向量放在一個連續矩陣、過濾欄位放整數陣列，過濾用向量化的遮罩，結果與 `exact_store.py` 相同；存檔在 `results/vector_db/<collection 名稱>/`（`meta.json`、`vectors.f32`、`records.jsonl`），新記憶附加寫入、內容改變時整份重寫，開啟時丟掉寫到一半的列；一次只能有一個寫入者 | 手動維護 | 是 |
| `chroma_store.py` | 檔案 | ChromaDB 內嵌模式（HNSW、cosine）：每個模型與維度一個 collection（`mem__<模型>__<維度>`），開啟時檢查模型與維度；`MemoryFilter` 轉成 Chroma 的 `where`（沒有日期的記憶存 `event_date_ord = 0`）；以 `memory_id` upsert，重跑不重複 | 手動維護 | 是 |
