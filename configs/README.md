# configs

存放設定檔，不含程式邏輯。模型資訊、路徑與門檻一律放這裡，不寫死在程式中。其他設定檔會隨任務加入，規劃見計畫書第 5 節。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `models.yaml` | 檔案 | 模型 registry：Hugging Face ID、固定的 revision、維度、prefix、精度、授權 | 手動維護，修改後要跑 `uv run pytest tests/contract` | 是 |

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
