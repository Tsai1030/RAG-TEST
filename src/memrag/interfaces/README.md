# interfaces

第 4 層。使用者入口，只呼叫 Facade（`memrag` 套件的公開 API），不直接操作 `application` 或 `infrastructure`。手動測試 UI 與報告產生器會在之後的任務加入。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `cli/` | 資料夾 | `memrag` 指令，說明見 [`cli/README.md`](cli/README.md) | — | 是 |
| `ui/` | 資料夾 | 手動測試 UI（Gradio），說明見 [`ui/README.md`](ui/README.md) | — | 是 |
