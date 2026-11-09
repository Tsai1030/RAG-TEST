# tests/unit/infrastructure

Infrastructure 層中不需模型或網路的部分。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_prompts.py` | 檔案 | `prompts/` 的每份 prompt 都讀得到、宣告的變數和內文的 `{{變數}}` 一致、能代入；找不到檔案或名稱不符時報錯 | 手動維護 | 是 |
| `test_vector_filters.py` | 檔案 | 過濾語意（日期條件排除沒有日期的記憶、多個類別任一符合）、轉成 Chroma `where` 的格式、collection 名稱、精確搜尋的排序與冪等寫入 | 手動維護 | 是 |
| `test_numpy_store.py` | 檔案 | `NumpyStore` 與精確基準 `ExactStore` 一致：220 組隨機過濾接受的記憶相同、各種 Top-k 的搜尋結果相同；冪等寫入、取回順序、空結果與維度錯誤；存檔後重新開啟結果相同、重跑不寫檔、內容改變時重寫、修復寫到一半的列、拒絕別的模型；`bootstrap` 依設定選向量庫 | 手動維護 | 是 |
| `test_probe_sampler.py` | 檔案 | 量測用的記憶體取樣執行緒能啟動、記錄並正常停止 | 手動維護 | 是 |
| `test_registry.py` | 檔案 | `models.yaml` 驗證：缺欄位、revision 不是完整 hash、未知欄位、候選都可商用、下載完整性判斷 | 手動維護 | 是 |
