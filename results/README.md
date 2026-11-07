# results

執行結果。`runs/` 由程式自動產生，不進版控；整理後的結論寫成 `reports/records/` 的紀錄。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容與格式 | 手動維護 | 是 |
| `runs/` | 資料夾 | 每次執行一個子資料夾 `<日期-時間>_<種類>/`，格式見下方 | `memrag models probe` 等指令產生；Nano 上的結果用 `scripts/nano_fetch.ps1` 取回 | 否 |
| `service_logs/` | 資料夾 | 只在 Nano 上：`scripts/nano_services.sh` 重啟常駐服務時，原本沒有記錄檔的服務會寫到這裡；`nano_llm_session.sh` 的 llama-server 輸出也在這裡 | 自動產生 | 否 |
| `vector_db/` | 資料夾 | `memrag store build` 建的 ChromaDB 向量庫，每個模型一個 collection；可隨時重建 | `memrag store build --model <模型>` | 否 |
| `llm_cache/` | 資料夾 | 只在 Nano 上：通過驗證的 LLM 總結輸出，一個 key 一個 JSON 檔；key 是對話、prompt、模型與參數的 hash，任何一項改變就不會命中 | `memrag summarize`、`memrag corpus build` 自動產生 | 否 |

## runs/<run_id>/ 的檔案

| 檔案 | 說明 |
|---|---|
| `manifest.json` | 執行種類、開始與結束時間、備註、模型 revision、主機、套件版本、git commit、開始時的可用記憶體 |
| `load_probe.jsonl` | `memrag models probe` 的結果，每個模型與裝置一行；欄位說明見 `src/memrag/infrastructure/embedding/README.md` |
| `metrics.json`、`per_query.jsonl`、`vectors/` | `memrag eval run` 的結果：各模型三種搜尋範圍的指標與信賴區間、逐題的前 10 名、語料與問題的向量（`eval rescore` 重算時使用） |
| `attempts.jsonl`、`failures.jsonl`、`memories.jsonl` | `memrag corpus build` 的結果：每場對話的每次嘗試與驗證錯誤、失敗的對話、這次產生的記憶 |
