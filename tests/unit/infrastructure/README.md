# tests/unit/infrastructure

Infrastructure 層中不需模型或網路的部分。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_probe_sampler.py` | 檔案 | 量測用的記憶體取樣執行緒能啟動、記錄並正常停止 | 手動維護 | 是 |
| `test_registry.py` | 檔案 | `models.yaml` 驗證：缺欄位、revision 不是完整 hash、未知欄位、候選都可商用、下載完整性判斷 | 手動維護 | 是 |
