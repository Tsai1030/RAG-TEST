# infrastructure/embedding

Embedding 模型相關實作：讀 registry、下載、載入與量測。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `registry.py` | 檔案 | 讀取並驗證 `configs/models.yaml`；找出已下載的固定 revision | 手動維護 | 是 |
| `download.py` | 檔案 | 把固定 revision 下載到專案的 `.hf_cache`，略過 ONNX 等不需要的檔案 | 手動維護 | 是 |
| `sentence_transformer.py` | 檔案 | `EmbeddingProvider` 的 Sentence Transformers 實作：統一加 prefix、截斷長度、輸出正規化向量；`batch_size` 可在載入後更改（量測不同 batch 用） | 手動維護 | 是 |
| `fake.py` | 檔案 | 測試用的確定性實作：同一段文字永遠得到同一個向量，不需載入模型 | 手動維護 | 是 |
| `probe.py` | 檔案 | 在獨立子行程載入模型，量測載入時間、記憶體峰值、單句延遲與超長輸入行為 | 手動維護 | 是 |

## 量測項目（probe.py）

| 欄位 | 說明 |
|---|---|
| `load_s` | 從建立模型到可以使用的秒數，不含 import |
| `latency_ms_p50`、`latency_ms_p95` | 暖機 3 次後，單句 Query 執行 20 次的延遲 |
| `batch32_ms` | 一次處理 32 筆記憶的時間 |
| `long_input_*` | 約 1,400 字的輸入：Token 數、是否截斷、耗時、輸出是否正常 |
| `sys_mem_peak_mib` | 量測期間系統可用記憶體的最大減少量；Nano 的 CPU 與 GPU 共用記憶體，這是主要指標（D1） |
| `cuda_peak_mib` | PyTorch GPU 配置峰值 |
| `process_peak_rss_mib` | 子行程的 RSS 峰值 |
