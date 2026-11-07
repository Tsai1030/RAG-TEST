# docs/adr

架構決策紀錄（Architecture Decision Record）。一個決定一個檔案，編號遞增；決定改變時新增一份並在舊的標註「已被取代」，不改寫舊內容。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容與規則 | 手動維護 | 是 |
| `0001-embedding-model.md` | 檔案 | 對話記憶的 Embedding 模型採用 ruri-v3-130m（備選 ruri-v3-30m），狀態：提議中 | 手動撰寫，依據 `docs/analysis/selection_report.md` | 是 |
