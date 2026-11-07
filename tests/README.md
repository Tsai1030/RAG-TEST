# tests

自動測試，策略見計畫書第 8 節。

```powershell
uv run pytest -q                     # 全部；沒有下載模型或沒有 GPU 的項目自動略過
uv run pytest -q tests/contract      # 修改 configs/models.yaml 後要跑
uv run pytest -q -m "not models"     # 只跑不需模型的測試
```

在 Windows 只會跑 Fake 實作；真實模型的契約測試要在 Nano 上跑（`.\scripts\nano_run.ps1 "uv run pytest -q"`）。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `architecture/` | 資料夾 | 分層依賴與 README 完整性，說明見 [`architecture/README.md`](architecture/README.md) | — | 是 |
| `unit/` | 資料夾 | 單元測試，說明見 [`unit/README.md`](unit/README.md) | — | 是 |
| `contract/` | 資料夾 | 同一個 Port 的所有實作都要通過的測試，說明見 [`contract/README.md`](contract/README.md) | — | 是 |
| `integration/` | 資料夾 | 真實套件與 CLI 的整合測試，說明見 [`integration/README.md`](integration/README.md) | — | 是 |
