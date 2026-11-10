# infrastructure

第 3 層。實作 `domain` 的 Port，並處理模型、向量庫、檔案等外部系統。只依賴 `domain`。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `memory_sampler.py` | 檔案 | 量測用的記憶體取樣：背景執行緒每 20 ms 讀系統 `MemAvailable` 記下最大降幅、行程峰值 RSS、PyTorch 的 CUDA 配置峰值（Embedding 載入、檢索與完整流程量測共用） | 手動維護 | 是 |
| `retrieval_probe.py` | 檔案 | 5.1.7 檢索量測的子行程：讀工作檔建立向量庫，記錄每題問題 Embedding、過濾＋搜尋、取回的耗時與記憶體，印出一行 JSON；另有只算向量的模式 | 手動維護 | 是 |
| `environment.py` | 檔案 | `memrag env check` 的各項檢查：Python、套件、GPU 實際運算、ChromaDB 讀寫、模型快取、記憶體與磁碟 | 手動維護 | 是 |
| `prompts.py` | 檔案 | `PromptStore` 實作：讀取 `prompts/<名稱>.v<版本>.yaml`，檢查名稱與版本和檔名一致；同一個 store 每個檔案只解析一次 | 手動維護 | 是 |
| `embedding/` | 資料夾 | Embedding 模型相關實作，說明見 [`embedding/README.md`](embedding/README.md) | — | 是 |
| `llm/` | 資料夾 | LLM 相關實作：llama-server 用戶端、測試用的假 LLM、回應快取，說明見 [`llm/README.md`](llm/README.md) | — | 是 |
| `vectorstore/` | 資料夾 | 向量庫：ChromaDB 與精確搜尋，說明見 [`vectorstore/README.md`](vectorstore/README.md) | — | 是 |
| `storage/` | 資料夾 | 執行結果的儲存，說明見 [`storage/README.md`](storage/README.md) | — | 是 |
