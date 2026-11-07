# 手動測試手冊

讓沒參與開發的人也能在 15 分鐘內完成冒煙測試。每次測試後新增一筆紀錄到 `reports/records/`，類型填「手動測試」。章節會隨任務增加。

## 事前準備

- **Windows：** 在專案根目錄執行 `uv sync`。
- **Nano：** 先 `.\scripts\nano_sync.ps1`，指令用 `.\scripts\nano_run.ps1 "<指令>"` 執行。真實模型只在 Nano 上有。

## 5.1.2 模型與環境

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag env check` | Nano 上全部 `OK`；Windows 上 `gpu` 可以是 `cuda_available: False`；`models` 在 Windows 顯示 `WARN` 是正常的 | |
| 2 | `uv run memrag models list` | 列出 5 個 `candidate` 和 1 個 `baseline`，每個都有 prefix；Nano 上都顯示「已下載」 | |
| 3 | `uv run memrag embed "テスト" --model ruri-v3-130m --kind query --explain` | `dim 512`、`norm 1.0`、`encoded_text '検索クエリ: テスト'` | |
| 4 | 同上，改成 `--kind document` | `encoded_text '検索文書: テスト'` | |
| 5 | `uv run memrag embed "テスト" --model qwen3-emb-0.6b --kind query --explain` | `dim 1024`；`encoded_text` 以 `Instruct:` 開頭，`Query:` 後面直接接文字，沒有空格 | |
| 6 | `uv run memrag embed "テスト" --model nope` | 顯示「錯誤（embed）」並列出可用模型，結束碼 2 | |
| 7 | Windows 上：`uv run memrag embed "テスト" --model ruri-v3-30m --explain --fake` | 不載入模型也能看到 prefix，`dim 256` | |

**常見問題**

- **第一次執行很慢：** 單次指令要 import PyTorch 並初始化 CUDA，Nano 上約 20–25 秒，這不是模型延遲。延遲請看載入量測的紀錄。
- **顯示未下載：** 在 Nano 上執行 `uv run memrag models download --models <名稱>`。
- **在 Nano 上記憶體不足：** 常駐服務佔約 4 GB，大模型請在量測時用 `scripts/nano_measure.sh` 停止服務後再測。

## 5.1.3 標準答案

不需要模型，Windows 上就能做。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag dataset validate` | 最後一行「結果：通過」；`personas 20`、`facts 400`、`queries 201`；`splits` 有 dev 和 test | |
| 2 | `uv run memrag dataset show-query q_p01_05` | 問題「ゆいちゃんがとまりにくるのはいつだったかしら。」；相關事實第一筆是 11 月 15 日孫の結衣ちゃんが泊まりに来る（2 分） | |
| 3 | `uv run memrag dataset show-query q_p01_10` | 無答案題，顯示「相關事實：無」 | |
| 4 | `uv run memrag dataset show-query q_p01_08` | 有日期過濾 `2026-10-25` 到 `2026-10-31`，相關事實的日期都在範圍內 | |
| 5 | `uv run memrag dataset agreement` | 印出 `kappa_all_pairs` 等數值與不一致的配對數 | |

**常見問題**

- **validate 出現錯誤：** 訊息會指出是哪一題或哪一筆事實，例如標了別人的事實、事實不符合題目的過濾條件。
- **想看日文是否自然：** 用 `show-query` 逐題看；審閱紀錄在 `docs/annotation_review.md`。

## 5.1.8 檢索準確度（預評估）

目前把 Gold Facts 當記憶庫。第 1 項 Windows 上就能做；第 2 項要在 Nano 上跑，Qwen3 用 float32 約需 3 GiB，要先停止常駐服務。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag eval run --models ruri-v3-30m,e5-small --fake` | 印出三段表格；「只搜本人」的 nDCG@10 約 0.2–0.3（假向量等於隨機，20 選 1）；「錯人」欄在只搜本人時是 0.00 | |
| 2 | Nano 上：`nohup bash scripts/nano_measure.sh "uv run memrag eval run --models all --split dev" > /tmp/rag_eval.log 2>&1 &` | 約 2 分鐘跑完並重啟服務；「只搜本人」時 ruri 系列的 nDCG@10 在 0.9 以上，bge-base-zh-v1.5 約 0.78 | |
| 3 | 打開第 2 項結果資料夾的 `per_query.jsonl`，找一題 `scope` 為 `person` 的紀錄 | `top` 列出前 10 筆的事實 ID、分數與標註分數；fact ID 的使用者編號都和題目相同 | |

**常見問題**

- **--fake 也產生了結果資料夾：** 資料夾名稱結尾是 `_fake`，可以直接刪除。
- **想用測試集：** `--split test` 只在最終評估使用，凍結門檻與閾值前不要跑。