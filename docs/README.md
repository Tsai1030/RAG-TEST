# docs

存放專案文件。其他文件會隨任務加入，規劃見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `project_plan.html` | 檔案 | 計畫書的網頁版，內容對應根目錄的 `PROJECT_PLAN.md`，也是線上網頁的原始檔；「測試結果」分頁的資料內嵌在檔案內 | 手動維護，要與 `PROJECT_PLAN.md` 一起更新；紀錄由 `scripts/build_results_tab.ps1` 寫入 | 是 |
| `manual_testing.md` | 檔案 | 手動測試手冊：各任務的指令、預期結果與常見問題 | 手動維護，每項任務完成時加一章 | 是 |
| `survey/` | 資料夾 | 候選模型調查，說明見 [`survey/README.md`](survey/README.md) | — | 是 |
