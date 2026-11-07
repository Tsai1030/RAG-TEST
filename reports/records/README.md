# reports/records

每次測試或模型調查一個 JSON 檔，對應網頁下拉選單的一個項目。紀錄規則見計畫書第 9 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容與格式 | 手動維護 | 是 |
| `<日期>_<主題>.json` | 檔案 | 一筆紀錄，檔名去掉 `.json` 就是紀錄 ID；數量等於已做過的測試與調查次數 | 手動新增，之後由 `memrag report add` 產生 | 是 |

## 紀錄格式

| 欄位 | 說明 |
|---|---|
| `title`、`date` | 主題與日期 |
| `task`、`type`、`status` | 所屬任務 5.1.x；類型；狀態為通過、失敗、部分完成、中止或資訊 |
| `backfilled` | `true` 表示事後補登 |
| `purpose` | 想回答什麼問題 |
| `environment` | 機器、版本等環境資訊清單 |
| `method` | 實際做法清單 |
| `results` | 結果區塊清單；`kind` 為 `p`、`list`、`table` 或 `note` |
| `conclusion`、`next` | 結論與下一步清單 |
| `sources` | 來源清單；`text` 必填，`url` 選填 |
| `related` | 相關紀錄清單；`id` 是紀錄 ID，`relation` 說明關係 |

舊紀錄不覆寫。發現錯誤時新增一筆紀錄，並在兩邊的 `related` 互相連結。
