# application

第 2 層。放 Use case（摘要、寫入、檢索、回答、評估、選型），只依賴 `domain` 的 Port，不得 import `infrastructure`。

5.1.3 加入評估資料的檢查；摘要、寫入、檢索等 Use case 會在 5.1.4 之後加入，規劃見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空；讓分層檢查找得到這一層 | 手動維護 | 是 |
| `evaluation/` | 資料夾 | 評估相關的 Use case，說明見 [`evaluation/README.md`](evaluation/README.md) | — | 是 |
