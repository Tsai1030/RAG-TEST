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
