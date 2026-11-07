# 手動測試手冊

讓沒參與開發的人也能在 15 分鐘內完成冒煙測試。每次測試後新增一筆紀錄到 `reports/records/`，類型填「手動測試」。章節會隨任務增加。

## 事前準備

- **Windows：** 在專案根目錄執行 `uv sync`。
- **Nano：** 先 `.\scripts\nano_sync.ps1`，指令用 `.\scripts\nano_run.ps1 "<指令>"` 執行。真實模型只在 Nano 上有。

## 5.1.2 模型與環境

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag env check` | Nano 上全部 `OK`；Windows 上 `gpu` 可以是 `cuda_available: False`；`models` 在 Windows 顯示 `WARN` 是正常的 | |
| 2 | `uv run memrag models list` | 列出 5 個 `candidate` 和 1 個 `baseline`，每個都有 prefix；Nano 上都顯示「已下載」 | |
| 3 | `uv run memrag embed "テスト" --model ruri-v3-130m --kind query --explain` | `dim 512`、`norm 1.0`、`encoded_text '検索クエリ: テスト'` | |
| 4 | 同上，改成 `--kind document` | `encoded_text '検索文書: テスト'` | |
| 5 | `uv run memrag embed "テスト" --model qwen3-emb-0.6b --kind query --explain` | `dim 1024`；`encoded_text` 以 `Instruct:` 開頭，`Query:` 後面直接接文字，沒有空格 | |
| 6 | `uv run memrag embed "テスト" --model nope` | 顯示「錯誤（embed）」並列出可用模型，結束碼 2 | |
| 7 | Windows 上：`uv run memrag embed "テスト" --model ruri-v3-30m --explain --fake` | 不載入模型也能看到 prefix，`dim 256` | |

**常見問題**

- **第一次執行很慢：** 單次指令要 import PyTorch 並初始化 CUDA，Nano 上約 20–25 秒，這不是模型延遲。延遲請看載入量測的紀錄。
- **顯示未下載：** 在 Nano 上執行 `uv run memrag models download --models <名稱>`。
- **在 Nano 上記憶體不足：** 常駐服務佔約 4 GB，大模型請在量測時用 `scripts/nano_measure.sh` 停止服務後再測。
