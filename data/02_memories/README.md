# data/02_memories

5.1.4 凍結的記憶語料：`01_dialogues/` 的 390 場對話，由 Qwen3-4B-Instruct-2507（Q4_K_M，在 Nano 上自己啟動的 llama-server）總結成的記憶。5.1.8 用 `memrag eval run --corpus memories` 評估。

| 項目 | 數值 |
|---|---|
| 記憶 | 641 筆（390 場對話全部成功，第一次就通過驗證，沒有重試） |
| Gold Fact 覆蓋率 | 98%（400 筆中 8 筆沒有任何記憶對應，清單在 `corpus_manifest.json`） |
| 沒有對應到 Gold Fact 的記憶 | 210 筆：多半是使用者在事實之後補的感想或細節，少數是閒聊 |
| `category` 與 Gold Fact 一致 | 75% |
| `event_date` 正確 | 有日期的事實 77%；沒有日期的事實 97% 正確填成 null |
| `content` 留著「昨日」「来月」等相對日期 | 103 筆（16%） |

這些錯誤是 4B 級本地模型的真實總結品質，評估時不修正：日期或類別錯了，「本人＋日期／類別過濾」就會漏掉正確的記憶。

**不要直接修改 `memories.jsonl`：** 它的 sha256 記在 `corpus_manifest.json` 與每次評估的 manifest，`memrag eval rescore` 會拒絕重算語料已經變動的評估。要重做語料時執行 `memrag corpus build`。

## 這個資料夾的檔案

| 名稱 | 類型 | 說明 | 產生方式 | 進版控 |
|---|---|---|---|---|
| `README.md` | 檔案 | 本文件：說明這個資料夾的內容 | 手動維護 | 是 |
| `memories.jsonl` | 檔案 | 一行一筆 `MemoryRecord`：LLM 填的 `category`、`content`、`event_date`、`people`、`source_turn_ids`，程式補的 ID 與日期，以及評估用的 `gold_fact_ids` | `memrag corpus build`（Nano，run_id `20261107-231513_summarize`） | 是 |
| `corpus_manifest.json` | 檔案 | 語料的 sha256、對話檔的 sha256、LLM 與參數、prompt 的 hash、驗收數字、未覆蓋的事實 | 同上 | 是 |
| `pool_labels.jsonl` | 檔案 | 池化審查（計畫 5.1.8）：開發集每題取各模型前 5 名（無答案題前 3 名），對沒有對應到事實或繼承了分數的 400 組記憶重新給 0／1／2 分，覆蓋繼承的分數 | Claude 依 `docs/annotation_guide.md` 判斷（annotator `claude-pool`） | 是 |

## 記憶的分數

記憶繼承它對應的 Gold Fact 的最高分。對應方式：記憶引用的句子（`source_turn_ids`）落在對話 `fact_turns` 裡哪筆事實的句子上。`pool_labels.jsonl` 有判斷的配對，以判斷為準；池化結果：35 組調高（例如「浩二の店にアジ二十匹を持っていく」對釣魚的問題給 1 分）、8 組調低（例如只寫「井上さんは授乳中です」的記憶，對「改喝無咖啡因咖啡」的問題從 2 分降為 1 分）。
