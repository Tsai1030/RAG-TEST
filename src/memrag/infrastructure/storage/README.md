# infrastructure/storage

執行結果的儲存。JSONL 資料讀寫會在 5.1.3 之後加入。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `manifest.py` | 檔案 | 建立 `results/runs/<run_id>/`，記錄主機、套件版本、git commit 與可用記憶體 | 手動維護 | 是 |
| `jsonl_repo.py` | 檔案 | 用 pydantic 讀寫 JSONL；格式錯誤時指出檔名與行號 | 手動維護 | 是 |
