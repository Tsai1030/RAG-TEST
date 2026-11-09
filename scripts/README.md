# scripts

存放操作用的薄包裝腳本，主要邏輯之後會放在 `src/`。其他規劃中的腳本見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `nano_sync.ps1` | 檔案 | 把 Windows 專案資料夾同步到 Nano 的 `~/50019/rag_test` | 手動維護 | 是 |
| `nano_run.ps1` | 檔案 | 在 Nano 的專案資料夾執行一個指令，例如 `.\scripts\nano_run.ps1 "uv run pytest -q"` | 手動維護 | 是 |
| `nano_fetch.ps1` | 檔案 | 把 Nano 的 `results/runs/<run_id>/` 取回 Windows | 手動維護 | 是 |
| `nano_services.sh` | 檔案 | 在 Nano 上執行：量測前停止、量測後重啟 `llama-server` 與 `pocket_tts_server`（D8） | 手動維護 | 是 |
| `nano_measure.sh` | 檔案 | 在 Nano 上執行一次量測：停服務、依序跑指令、結束時一定重啟服務；用 `nohup` 背景執行，SSH 斷線也不影響 | 手動維護 | 是 |
| `nano_llm_session.sh` | 檔案 | 在 Nano 上執行一次總結：停服務、用 `configs/pipeline.yaml` 的模型啟動自己的 `llama-server`（:8095）、依序跑指令、結束時一定關掉它並重啟服務 | 手動維護 | 是 |
| `build_results_tab.ps1` | 檔案 | 把 `reports/` 的紀錄寫入計畫書網頁的「測試結果」分頁，`memrag report build` 完成後移除 | 手動維護 | 是 |

## nano_sync.ps1

在專案根目錄執行，連線時會詢問 SSH 密碼；腳本不儲存密碼。

```powershell
.\scripts\nano_sync.ps1
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `-Target` | `robo@172.28.0.147` | 連線對象 |
| `-RemoteDir` | `50019/rag_test` | 目的資料夾，相對於家目錄 |

- **不同步的資料夾：** `.venv`、`__pycache__`、`.pytest_cache`、`.ruff_cache`、`.mypy_cache`、`.import_linter_cache`、`.hf_cache`。`.git` 會同步，讓 Nano 上的執行結果能記錄 commit。
- **不會刪除 Nano 上的檔案：** 只新增和覆寫，所以 Windows 上刪除或改名的檔案要另外到 Nano 清理。

## nano_services.sh

在 Nano 的專案資料夾執行。量測前後各跑一次，不要只停不啟。

```bash
bash scripts/nano_services.sh status   # 查看兩個服務與可用記憶體
bash scripts/nano_services.sh stop     # 記下原本的指令與工作目錄後結束行程
bash scripts/nano_services.sh start    # 用同樣的指令重啟，等到服務可以連線
```

- **為什麼結束行程：** `SIGSTOP` 只會暫停，不會釋放記憶體。
- **只重啟自己停的服務：** 狀態存在 `~/50019/.service_state/`，沒有被這個腳本停止的服務不會被啟動。
- **確認方式：** `llama-server` 的 `/health` 回 200；`pocket_tts_server` 在 8092 埠監聽。

## nano_measure.sh

量測時用這個腳本，不要手動停服務：指令失敗或中斷時也會重啟服務。每個參數是一個依序執行的指令。

```bash
nohup bash scripts/nano_measure.sh "uv run memrag models probe" "uv run pytest -q" > /tmp/rag_measure.log 2>&1 &
```

## nano_llm_session.sh

和 `nano_measure.sh` 一樣會停服務並保證重啟，另外在 :8095 啟動總結用的 LLM（不共用 :8090 的服務）。伺服器的輸出寫到 `results/service_logs/rag_llm_server.log`。

```bash
nohup bash scripts/nano_llm_session.sh "uv run memrag corpus build" > /tmp/rag_llm.log 2>&1 &
# 問答階段：context 用 4096、Embedding 用 CPU，省記憶體
LLM_CTX=4096 nohup bash scripts/nano_llm_session.sh "uv run memrag eval answers --split dev --device cpu" > /tmp/rag_llm.log 2>&1 &
```

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `LLM_CTX` | `pipeline.yaml` 的 `context`（8192） | LLM 的 context 長度；問答只需要 4096，KV cache 約省 0.6 GB |
| `MIN_MEM_MB` | `500` | 記憶體保護：可用記憶體低於這個值時停掉正在跑的指令並結束這次階段（照樣重啟服務） |

- **為什麼要記憶體保護：** Nano 的 8 GB 由 CPU 與 GPU 共用，用完時會整台卡住好幾個小時，系統才砍掉程序（2026-11-08 12:14 卡住，15:55 才由 OOM killer 砍掉 llama-server），所以在卡住之前主動停止。
- **`--cache-ram 0`：** 這版 llama-server 預設會把處理過的 prompt 的 KV 狀態存在主記憶體，上限 8192 MiB，比 Nano 的記憶體還大。問答評估每題的 prompt 都不同，會一直累積到記憶體用完（2026-11-09 評估開始兩分鐘就觸發記憶體保護），所以關掉。重複的 prompt 由 memrag 自己的回應快取處理。
- **紀錄：** 每個指令結束時印出 `== exit <代碼> (MemAvailable min <MiB>)`；因記憶體不足停止時印出 `session ended: low memory`，結束碼 3。
