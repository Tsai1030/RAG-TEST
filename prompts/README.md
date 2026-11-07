# prompts

存放所有送給 LLM 的 prompt。`src/` 不寫 prompt 內文，只用名稱和版本讀取：

```python
prompt = prompt_store.get("summarize", version="v1")
```

讀取程式 `PromptStore` 尚未實作，規劃見計畫書第 4、5 節。目前四份都是草稿（`status: draft`），還沒有用真實 LLM 測過。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容與規則 | 手動維護 | 是 |
| `dialogue_generate.v1.yaml` | 檔案 | 依人物設定和 Gold Facts 生成假對話（5.1.4） | 手動維護 | 是 |
| `summarize.v1.yaml` | 檔案 | 把對話紀錄轉成記憶 JSON 清單（5.1.4） | 手動維護 | 是 |
| `summarize_retry.v1.yaml` | 檔案 | 摘要驗證失敗時，接在 `summarize` 訊息後面要求修正（5.1.4） | 手動維護 | 是 |
| `answer.v1.yaml` | 檔案 | 依檢索到的記憶回答問題並附引用（5.1.6） | 手動維護 | 是 |

## 檔案格式

檔名為 `<名稱>.v<版本>.yaml`，內容欄位如下：

| 欄位 | 說明 |
|---|---|
| `name`、`version` | 必須與檔名一致 |
| `status` | `draft`（可修改）或 `frozen`（已凍結） |
| `purpose` | 這份 prompt 的用途 |
| `used_by` | 使用它的程式 |
| `task` | 對應的計畫書任務 |
| `variables` | 內文用到的變數，呼叫時缺少任何一個都要報錯 |
| `system` | System 訊息；`null` 表示接在前一段訊息後面，不另外送 |
| `user` | User 訊息，用 `{{變數}}` 標示要代入的位置 |

`output_schema` 由程式從 Pydantic Schema 產生後代入，不在 prompt 裡手寫欄位定義。`refusal_text` 來自 `configs/pipeline.yaml`，與程式直接拒答時用的文字相同。

## 規則

- **語言：** 這輪只做日文，`system` 和 `user` 的指令與模型輸出都用日文。`purpose` 這類給人看的欄位用繁中。
- **版本：** `draft` 可以直接修改。改成 `frozen` 後內容不再變動，要修改就新增版本，例如 `summarize.v2.yaml`。
- **測試目的：** 這輪的比較對象是 Embedding 模型，prompt 只是固定條件，不比較版本，也不找最佳 prompt。只要流程跑得通、語料達到 5.1.4 驗收就凍結。
- **不放這裡的內容：** Embedding 模型的 query／document prefix 屬於模型規格，放在 `configs/models.yaml`。
