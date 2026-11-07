# 對話記憶 RAG：Embedding 模型評估與選型

以日文對話記憶情境，評估並選出最適合的 Embedding 模型。

**目前狀態：** 5.1.1 模型調查已完成，見 [`docs/survey/model_survey.md`](docs/survey/model_survey.md)。5.1.2 已完成：專案骨架、模型 registry、`memrag` 指令與 Nano 上的載入量測；首輪名單為 5 個候選加 1 個對照組，見 `configs/models.yaml`。完整內容見 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)，規劃中的資料夾結構見計畫書第 5 節。

## 快速開始

```powershell
uv sync                                                    # 建立環境（Windows 用 CPU 版 PyTorch）
uv run pytest -q                                           # 自動測試
uv run memrag models list                                  # 候選模型與 prefix
uv run memrag embed "テスト" --model ruri-v3-30m --explain --fake  # 不載入模型，只看 prefix
```

在 Nano 上的指令：

```powershell
.\scripts\nano_sync.ps1
.\scripts\nano_run.ps1 "uv sync && uv run memrag env check"
.\scripts\nano_run.ps1 "uv run memrag embed 'テスト' --model ruri-v3-130m --kind query --explain"
```

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：專案說明與資料夾導覽 | 手動維護 | 是 |
| `PROJECT_PLAN.md` | 檔案 | 計畫書：任務、架構、測試策略與決策紀錄 | 手動維護 | 是 |
| `pyproject.toml` | 檔案 | 依賴、`memrag` 指令入口、pytest 與分層規則（import-linter）設定 | 手動維護 | 是 |
| `uv.lock` | 檔案 | 鎖定所有平台的依賴版本 | `uv lock` 產生 | 是 |
| `.gitignore` | 檔案 | 不進版控的檔案規則 | 手動維護 | 是 |
| `.gitattributes` | 檔案 | 讓 `.sh` 腳本在 Windows 上也保持 LF 換行，同步到 Nano 才能執行 | 手動維護 | 是 |
| `configs/` | 資料夾 | 設定檔，說明見 [`configs/README.md`](configs/README.md) | — | 是 |
| `docs/` | 資料夾 | 專案文件，說明見 [`docs/README.md`](docs/README.md) | — | 是 |
| `prompts/` | 資料夾 | 所有 LLM prompt，說明見 [`prompts/README.md`](prompts/README.md) | — | 是 |
| `reports/` | 資料夾 | 測試與模型調查的紀錄，說明見 [`reports/README.md`](reports/README.md) | — | 是 |
| `results/` | 資料夾 | 執行結果，說明見 [`results/README.md`](results/README.md) | — | 只有 README |
| `scripts/` | 資料夾 | 操作腳本，說明見 [`scripts/README.md`](scripts/README.md) | — | 是 |
| `src/` | 資料夾 | Python 原始碼，說明見 [`src/README.md`](src/README.md) | — | 是 |
| `tests/` | 資料夾 | 自動測試，說明見 [`tests/README.md`](tests/README.md) | — | 是 |

`.venv/`（Python 環境）與 `.hf_cache/`（模型檔，只在 Nano 上）不需要 README，也不進版控。

## 開發與執行位置

- **Windows `D:\RAG TEST`：** 寫程式和跑不需要 GPU 的快速測試。
- **Jetson Nano `~/50019/rag_test`：** 執行所有 GPU 工作。

修改後要把資料同步到 Nano：

```powershell
.\scripts\nano_sync.ps1
```

## README 規則

每個資料夾都要有 `README.md`，說明資料夾用途並列出裡面的每個檔案和子資料夾。完整規則見計畫書第 5 節。
