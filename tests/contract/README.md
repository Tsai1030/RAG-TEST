# tests/contract

同一個 Port 的所有實作都要通過同一組測試。真實模型只在已下載的機器（Nano）上跑，其他機器只測 Fake。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_embedding_provider.py` | 檔案 | `EmbeddingProvider` 契約與 Prefix 契約：prefix 與 model card 一致、與模型內建 prompt 一致、真的加在送進模型的文字前面；輸出維度、單位長度、可重現、批次與單筆一致 | 手動維護 | 是 |
