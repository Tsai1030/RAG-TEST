# tests/unit/domain

Domain 層的單元測試。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `test_model_spec.py` | 檔案 | prefix 規則：query／document 各加對的 prefix、空 prefix 不改文字、未知 kind 報錯 | 手動維護 | 是 |
| `test_memory_schema.py` | 檔案 | 記憶 Schema：`content` 必須是日文且不超過 200 字、LLM 不能填 `person_id`、程式補上的 ID 與日期 | 手動維護 | 是 |
| `test_statistics.py` | 檔案 | Cohen's κ 與手算結果一致、邊界情況 | 手動維護 | 是 |
| `test_dialogue_schema.py` | 檔案 | 對話紀錄：送給 LLM 的格式不含事實標記、句子編號要連續、`fact_turns` 指到存在的句子；`PromptTemplate` 缺少或多出變數時報錯 | 手動維護 | 是 |
| `test_retrieval_metrics.py` | 檔案 | Recall@k、MRR、nDCG（分級）與手算結果一致；AUC、Bootstrap 與配對 Bootstrap 信賴區間 | 手動維護 | 是 |
