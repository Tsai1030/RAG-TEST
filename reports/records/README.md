# reports/records

實驗結果，一次實驗一個 JSON 檔，對應網頁下拉選單的一個項目。紀錄規則見計畫書第 9 節。

**放什麼：** 模型調查與篩選、模型效能量測、檢索指標評估、選型分析，也就是週會要報告的結果。環境檢查、開發中的測試、下載、修 bug 這些過程不放；實驗中途失敗或有重要問題，寫在該次實驗紀錄的結果或注意事項裡。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容與格式 | 手動維護 | 是 |
| `<日期>_<主題>.json` | 檔案 | 一筆實驗結果，檔名去掉 `.json` 就是紀錄 ID；數量等於已做過的實驗次數 | 手動新增，之後由 `memrag report add` 產生 | 是 |

## 紀錄格式

| 欄位 | 說明 |
|---|---|
| `title`、`date` | 主題與日期 |
| `task`、`type`、`status` | 所屬任務 5.1.x；類型為模型調查、模型篩選、效能量測、指標評估或選型分析；狀態為通過、失敗、部分完成、中止或資訊 |
| `backfilled` | `true` 表示事後補登 |
| `purpose` | 這次實驗想回答什麼問題 |
| `environment` | 機器、版本、模型 revision 等環境資訊清單 |
| `method` | 實際做法清單 |
| `results` | 結果區塊清單；`kind` 為 `p`、`list`、`table`、`note` 或 `chart` |

`chart` 區塊在網頁上畫成 SVG 圖，`overview.json` 的 `charts` 也用同一個格式：

| 欄位 | 說明 |
|---|---|
| `chart` | `dot`（點＋信賴區間，軸可不從 0 開始，用於差距小的比較）、`bar`（長條，軸從 0 開始）或 `scatter`（散佈圖） |
| `caption`、`note` | 圖的標題與說明；標題寫清楚越高或越低越好 |
| `categories`、`series` | `dot`／`bar` 用：每列一個類別；`series` 每項有 `name`、`values`，可加 `ci`（每個值的 `[下限, 上限]`） |
| `min`、`max`、`digits`、`axisDigits` | 橫軸範圍、數值與刻度的小數位數 |
| `x`、`y`、`points`、`legend` | `scatter` 用：軸的 `min`、`max`、`digits`、`label`；每個點有 `label`、`x`、`y`，`group` 決定顏色，`dy` 微調標籤位置 |
| `conclusion`、`next` | 結論與下一步清單 |
| `sources` | 來源清單；`text` 必填，`url` 選填 |
| `related` | 相關紀錄清單；`id` 是紀錄 ID，`relation` 說明關係；指向 `log/` 的紀錄不會顯示在網頁上 |

舊紀錄不覆寫。發現錯誤時新增一筆紀錄，並在兩邊的 `related` 互相連結。
