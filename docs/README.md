# docs

存放專案文件。其他文件會隨任務加入，規劃見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `project_plan.html` | 檔案 | 計畫書的網頁版，內容對應根目錄的 `PROJECT_PLAN.md`，也是線上網頁的原始檔；「測試結果」分頁的資料內嵌在檔案內 | 手動維護，要與 `PROJECT_PLAN.md` 一起更新；紀錄由 `scripts/build_results_tab.ps1` 寫入 | 是 |
| `manual_testing.md` | 檔案 | 手動測試手冊：各任務的指令、預期結果與常見問題 | 手動維護，每項任務完成時加一章 | 是 |
| `annotation_guide.md` | 檔案 | 標註指南：測試問題對應 Gold Fact 的評分規則（0／1／2），第二次標註也照這份 | 手動維護 | 是 |
| `annotation_review.md` | 檔案 | 5.1.3 兩次標註的一致度（Cohen's κ）、分歧裁決與日文審閱的處理紀錄 | 手動維護 | 是 |
| `analysis/` | 資料夾 | 5.1.9 的選型報告，說明見 [`analysis/README.md`](analysis/README.md) | — | 是 |
| `adr/` | 資料夾 | 架構決策紀錄（ADR），說明見 [`adr/README.md`](adr/README.md) | — | 是 |
| `survey/` | 資料夾 | 候選模型調查，說明見 [`survey/README.md`](survey/README.md) | — | 是 |
| `schema/` | 資料夾 | 資料格式說明，見 [`schema/README.md`](schema/README.md) | — | 是 |
