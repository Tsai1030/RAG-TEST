# application

第 2 層。放 Use case（摘要、寫入、檢索、回答、評估、選型），只依賴 `domain` 的 Port，不得 import `infrastructure`。

5.1.3 加入評估資料的檢查，5.1.4 加入對話總結，5.1.6 加入寫入與問答。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空；讓分層檢查找得到這一層 | 手動維護 | 是 |
| `summarize.py` | 檔案 | 對話 → 記憶：結構化輸出、Schema 與語意驗證、附錯誤訊息重試最多 3 次、快取、依引用的對話句子對應到 Gold Fact | 手動維護 | 是 |
| `llm_calls.py` | 檔案 | 所有 LLM 呼叫共用的快取 key（訊息、模型與取樣參數、輸出 Schema 的 hash） | 手動維護 | 是 |
| `ingest.py` | 檔案 | 對話 → 向量庫（5.1.6）：總結、向量化、寫入；總結重試後仍失敗就不寫入，回報各階段耗時 | 手動維護 | 是 |
| `answer.py` | 檔案 | 問答（5.1.6）：過濾檢索 Top-k → 閾值（第一層，沒有通過就不呼叫 LLM；`per_memory` 只送通過的記憶，`gate` 第一名通過就送全部 Top-k）→ Context Build（去重、依分數排序、超過字數預算從最低分捨棄）→ LLM 回答（可拒答，第二層）→ 檢查引用是否都在 Context 裡；附各階段耗時 | 手動維護 | 是 |
| `evaluation/` | 資料夾 | 評估相關的 Use case，說明見 [`evaluation/README.md`](evaluation/README.md) | — | 是 |
