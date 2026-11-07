# interfaces/cli/commands

每個檔案是一組子指令，提供 `register()` 註冊參數。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `env.py` | 檔案 | `memrag env check`：環境檢查；在 Nano 上沒有 GPU 就算失敗 | 手動維護 | 是 |
| `models.py` | 檔案 | `memrag models list`、`download`、`probe`：列出、下載與量測模型 | 手動維護 | 是 |
| `embed.py` | 檔案 | `memrag embed`：把一句文字轉成向量；`--explain` 顯示實際送進模型的文字 | 手動維護 | 是 |
