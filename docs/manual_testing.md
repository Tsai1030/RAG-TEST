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

## 5.1.4 模擬對話與記憶總結

第 1–3 項 Windows 上就能做；第 4–5 項要在 Nano 上跑，會暫停常駐服務。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag dialogues validate` | `對話 390 場`，最後一行「結果：通過」 | |
| 2 | `uv run memrag summarize d_p01_1027 --fake` | 不呼叫 LLM，每句使用者發言變成一筆記憶；第 2 句對應 `f_p01_14`、第 5 句對應 `f_p01_17` | |
| 3 | `uv run memrag summarize d_p01_1027 --show-retries` | Windows 上沒有快取的話會連不到 LLM，顯示錯誤並結束；Nano 上全量執行後有快取，會顯示「（快取）」與 2 筆以上記憶 | |
| 4 | Nano 上：`nohup bash scripts/nano_llm_session.sh "uv run memrag corpus build --limit 17" > /tmp/rag_llm.log 2>&1 &` | 約 2 分鐘；`success_rate 1.0`；結尾「已凍結：否」；服務重啟 | |
| 5 | Nano 上：`bash scripts/nano_services.sh status` | 兩個服務都是 `running` | |

**常見問題**

- **改了 prompt 或 `configs/pipeline.yaml` 後結果沒變：** 快取的 key 包含 prompt 全文、模型與參數，有改就會重新呼叫；如果還是一樣，確認改的是 `prompt_version` 指向的那份檔案。
- **記憶沒有對應到 Gold Fact：** 用 `memrag summarize <log_id>` 看 `turns [...] → ...`；LLM 引用的句子沒有落在事實的句子上時就對應不到，這會算進「未覆蓋」。

## 5.1.5 向量庫與過濾檢索

第 1 項 Windows 上就能做（要先用 `nano_fetch.ps1` 取回一次記憶庫評估）；第 2–4 項在 Nano 上，不需要停服務。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag store check results/runs/20261108-005438_retrieval_eval_memories` | 6 個模型的「過濾不一致」與「跨人」都是 0；Recall@10「本人」與「過濾」是 1.000 | |
| 2 | `uv run memrag store build --model ruri-v3-130m` | `641 筆`；再跑一次筆數不變 | |
| 3 | `uv run memrag search "先週何かあったっけ" --model ruri-v3-130m --person p06 --from 2026-10-25 --to 2026-10-31 -k 3` | 只有 2 筆：10/25 走完十公里、10/30 第一次專案會議 | |
| 4 | `uv run memrag search "ラーメン" --model ruri-v3-130m --person p99` | 「沒有符合過濾條件的記憶」，不報錯 | |

**常見問題**

- **Windows 上 `store check` 結束後刪不掉暫存資料夾：** ChromaDB 在 Windows 會延後釋放檔案，程式已忽略這個錯誤，暫存資料夾可以手動刪除。
- **想清空向量庫：** 刪除 `results/vector_db/` 後重新 `store build`。

## 5.1.6 完整問答流程

第 1–4 項 Windows 上就能做（假模型、假 LLM）；第 5–8 項在 Nano 上，要用 `scripts/nano_llm_session.sh` 暫停常駐服務並啟動 LLM。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag ask "ラーメン" --person p06 --fake --trace` | 「関連する情報はありません」；判斷是「沒有記憶通過閾值，未呼叫 LLM（第一層）」；過濾顯示 `date_fallback: True`；檢索結果都是 p06 | |
| 2 | 同上，加 `--no-threshold` | 回答第一筆記憶的內容並引用它（假 LLM 的行為）；閾值顯示「未校準，不套用」 | |
| 3 | `uv run memrag eval answers --fake --limit 10` | 三種做法各一列（`threshold+llm` 逐筆閾值、`gate+llm` 閾值當閘門、`llm_only` 不用閾值）；「無效引用」與「過濾外」都是 0；「重跑檢索」結果不同 0 題 | |
| 4 | `uv run memrag ui --fake`，瀏覽器開 `http://127.0.0.1:7860` | 六個分頁都能操作；Ask 分頁顯示回答、判斷、檢索結果表與 Trace | |
| 5 | Nano 上：`LLM_CTX=4096 nohup bash scripts/nano_llm_session.sh "uv run memrag store build --model ruri-v3-130m" "uv run memrag ask 'とんこつラーメン、今もよく食べてるんだっけ？' --person p06 --today 2026-11-01 --trace --device cpu" > /tmp/rag_llm.log 2>&1 &` | 回答根據 10/24 的新記憶（月に二回まで），引用的 memory_id 都在 Context 裡；服務重啟 | |
| 6 | 同上，問題換成 `'釣りに行く予定ってあったっけ。' --person p02` | 拒答（閾值擋下或 LLM 拒答），沒有引用 | |
| 7 | 同上，問題換成 `'明日の天気はどうなるかな？' --person p06` | 拒答，不呼叫 LLM（第一層） | |
| 8 | Nano 上：`bash scripts/nano_services.sh status` | 兩個服務都是 `running` | |

**常見問題**

- **Nano 卡住、ssh 連不上：** 多半是記憶體用完。問答時加 `LLM_CTX=4096` 與 `--device cpu`；`nano_llm_session.sh` 在可用記憶體低於 500 MiB 時會自動停止並重啟服務。
- **ask 顯示 `stage answer` 的錯誤：** LLM 伺服器沒有啟動，`memrag ask` 只能在 `nano_llm_session.sh` 裡執行（或自行啟動 :8095）。
- **說沒有索引：** 先 `memrag store build --model <模型>`；5.1.6 起向量庫存了完整記錄，舊的索引要重建一次。
- **類別過濾沒有作用：** `ask` 依 5.1.6 的決定不使用類別過濾，日期過濾在記憶沒有事件日期時改看記錄日；`search` 則照輸入的條件過濾。

## 5.1.8 檢索準確度（預評估）

目前把 Gold Facts 當記憶庫。第 1 項 Windows 上就能做；第 2 項要在 Nano 上跑，Qwen3 用 float32 約需 3 GiB，要先停止常駐服務。

| # | 指令 | 預期結果 | 實際結果 |
|---|---|---|---|
| 1 | `uv run memrag eval run --models ruri-v3-30m,e5-small --fake` | 印出三段表格；「只搜本人」的 nDCG@10 約 0.2–0.3（假向量等於隨機，20 選 1）；「錯人」欄在只搜本人時是 0.00 | |
| 2 | Nano 上：`nohup bash scripts/nano_measure.sh "uv run memrag eval run --models all --split dev" > /tmp/rag_eval.log 2>&1 &` | 約 2 分鐘跑完並重啟服務；「只搜本人」時 ruri 系列的 nDCG@10 在 0.9 以上，bge-base-zh-v1.5 約 0.78 | |
| 3 | 打開第 2 項結果資料夾的 `per_query.jsonl`，找一題 `scope` 為 `person` 的紀錄 | `top` 列出前 10 筆的事實 ID、分數與標註分數；fact ID 的使用者編號都和題目相同 | |
| 4 | Nano 上：同第 2 項，指令加 `--corpus memories` | 改用 `data/02_memories` 的 641 筆記憶；「只搜本人」時 ruri 系列約 0.85–0.90 | |
| 5 | Windows 上：`uv run memrag eval rescore results/runs/20261107-235108_retrieval_eval_memories`（先用 `nano_fetch.ps1` 取回） | 不載入模型，約 1 分鐘；套用 `pool_labels.jsonl` 後 ruri-v3-310m 約 0.90 | |

**常見問題**

- **--fake 也產生了結果資料夾：** 資料夾名稱結尾是 `_fake`，可以直接刪除。
- **想用測試集：** `--split test` 只在最終評估使用，凍結門檻與閾值前不要跑。