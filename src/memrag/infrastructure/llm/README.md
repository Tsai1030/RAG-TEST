# infrastructure/llm

`LLMProvider` 與 `ResponseCache` 的實作，用於 5.1.4 的對話總結。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `__init__.py` | 檔案 | 套件標記，內容為空 | 手動維護 | 是 |
| `llama_server.py` | 檔案 | 呼叫 llama.cpp `llama-server` 的 OpenAI 相容 API：用 JSON Schema 限制輸出、關閉 Qwen3 的思考模式、網路錯誤以指數退避重試；只用標準函式庫 | 手動維護 | 是 |
| `fake.py` | 檔案 | 測試與 `--fake` 用的假 LLM：給定回應就依序回傳；沒給就把每句使用者發言當成一筆記憶 | 手動維護 | 是 |
| `cache.py` | 檔案 | 檔案式回應快取，一個 key 一個 JSON 檔；只存通過驗證的輸出 | 手動維護 | 是 |
