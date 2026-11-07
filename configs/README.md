# configs

存放設定檔，不含程式邏輯。模型資訊、路徑與門檻一律放這裡，不寫死在程式中。其他設定檔會隨任務加入，規劃見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `models.yaml` | 檔案 | 模型 registry：Hugging Face ID、固定的 revision、維度、prefix、精度、授權 | 手動維護，修改後要跑 `uv run pytest tests/contract` | 是 |
| `selection.yaml` | 檔案 | 選型規則（計畫 5.1.9）：測試集用哪個記憶庫與搜尋範圍、硬性門檻、加權方式與權重、平手規則；測試集評估前凍結 | 手動維護，2026-11-08 凍結；之後要改就升版並在報告註明 | 是 |
| `thresholds.yaml` | 檔案 | 各模型的分數閾值，以及校準用的語料與標註 hash、抽樣設定；評估測試集前要先提交 | `memrag eval calibrate <開發集的記憶庫評估> --write` 產生，不要手動修改 | 是 |
| `pipeline.yaml` | 檔案 | 流程設定：總結用 LLM（自己的 llama-server 位址、GGUF 的 revision、temperature、seed）、prompt 版本、快取位置、拒答文字 | 手動維護；改 LLM 或參數後快取自動失效 | 是 |

## models.yaml 欄位

| 欄位 | 說明 |
|---|---|
| `hf_id`、`revision` | Hugging Face 模型 ID 與固定的 commit hash |
| `role` | `candidate` 參與選型；`baseline` 只當對照組 |
| `dim`、`max_seq_length` | 向量維度；輸入截斷長度（Token 數） |
| `query_prefix`、`document_prefix` | 加在查詢與記憶前面的文字，依 model card |
| `dtype` | CUDA 載入精度；CPU 一律 `float32` |
| `trust_remote_code` | 是否執行模型附帶的程式碼；要先審閱遠端程式碼才可設為 `true` |
| `license`、`commercial_ok` | 授權與能否商用（D3：會商用） |
| `download_ignore` | 這個模型額外不下載的檔案 |
