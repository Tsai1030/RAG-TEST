# 5.1.1 候選 Embedding 模型調查

查閱日：2026-11-07。這份調查只用公開資料，**還沒有在 Nano 上載入或量測任何模型**。

## 結論

- **建議首輪 6 個（權重合計約 3.84 GiB）：** `ruri-v3-30m`、`ruri-v3-130m`、`ruri-v3-310m`、`GLuCoSE-base-ja-v2`、`Qwen3-Embedding-0.6B`、`multilingual-e5-small`。最後名單由 5.1.2 依實際載入結果決定。
- **備選：** `EmbeddingGemma`、`Nemotron-3-Embed-1B`、`PLaMo-Embedding-1B`。
- **計畫書修正：** PLaMo 的權重實際約 2.0 GiB，不是原本估計的 4 GB，不再因記憶體門檻排除。
- **新發現：** 2026 年有 Nemotron-3-Embed 和 LFM2.5-Embedding 兩個新模型，但都沒有 JMTEB 資料集的公開分數，無法直接比較。

公開分數都來自文件檢索資料集，不是對話記憶。它們只用來篩選，最終仍以 5.1.8 的測試為準。

## 候選規格

權重大小是 Hugging Face API 回傳的實際檔案大小。`trust_remote_code` 欄的「需要」表示模型會執行儲存庫內的程式碼。

| 模型 | 類型 | 參數量 | 維度 | 最大長度 | 權重 | 授權 | 商用 | trust_remote_code |
|---|---|---:|---:|---:|---:|---|---|---|
| `cl-nagoya/ruri-v3-30m` | 日文 | 37M | 256 | 8192 | 140 MiB | Apache 2.0 | 可 | 不需要 |
| `cl-nagoya/ruri-v3-70m` | 日文 | 70M | 384 | 8192 | 267 MiB | Apache 2.0 | 可 | 不需要 |
| `cl-nagoya/ruri-v3-130m` | 日文 | 132M | 512 | 8192 | 504 MiB | Apache 2.0 | 可 | 不需要 |
| `cl-nagoya/ruri-v3-310m` | 日文 | 315M | 768 | 8192 | 1,200 MiB | Apache 2.0 | 可 | 不需要 |
| `pkshatech/GLuCoSE-base-ja-v2` | 日文 | 133M | 768 | 512 | 508 MiB | Apache 2.0 | 可 | 不需要 |
| `pkshatech/RoSEtta-base-ja` | 日文 | 約 0.2B | 768 | 1024 | 726 MiB | Apache 2.0 | 可 | 需要 |
| `pfnet/plamo-embedding-1b` | 日文 | 1.05B | 2048 | 4096 | 2,004 MiB（BF16） | Apache 2.0 | 可 | 需要 |
| `sbintuitions/sarashina-embedding-v1-1b` | 日文 | 1.22B | 1792 | 8192 | 4,669 MiB | Sarashina 非商用 | **否** | 不需要 |
| `sbintuitions/sarashina-embedding-v2-1b` | 日文 | 約 1.2B | 1792 | 8192 | 4,669 MiB | Sarashina 非商用 | **否** | 不需要 |
| `intfloat/multilingual-e5-small` | 多語 | 118M | 384 | 512 | 449 MiB | MIT | 可 | 不需要 |
| `BAAI/bge-m3` | 多語 | 未標示 | 1024 | 8192 | 2,166 MiB | MIT | 可 | 不需要 |
| `google/embeddinggemma-300m` | 多語 | 約 0.3B | 768（可截斷） | 2048 | 1,155 MiB | Gemma，需同意條款 | 依條款 | 不需要 |
| `Qwen/Qwen3-Embedding-0.6B` | 多語 | 約 0.6B | 1024（可截斷） | 32K | 1,136 MiB | Apache 2.0 | 可 | 不需要 |
| `nvidia/Nemotron-3-Embed-1B-BF16` | 多語 | 1.14B | 2048（可截斷） | 32768 | 2,176 MiB（BF16） | OpenMDW-1.1 | 可 | 模型卡未提及 |
| `LiquidAI/LFM2.5-Embedding-350M` | 多語 | 354M | 1024 | 512 | 676 MiB（BF16） | LFM Open v1.0 | **有條件** | 需要 |

LFM Open License v1.0 只允許年營收低於 1,000 萬美元的單位商用。這裡的授權整理不是法律意見，正式商用前要再確認。

## Query 與 Document 規則

| 模型 | Query | Document | 依賴與注意事項 |
|---|---|---|---|
| ruri-v3（4 個） | `検索クエリ: ` | `検索文書: ` | `transformers>=4.48` |
| GLuCoSE v2 | `query: ` | `passage: ` | Tokenizer 是 `MLukeTokenizer`，5.1.2 要確認依賴 |
| RoSEtta | `query: ` | `passage: ` | 要執行遠端程式碼 |
| PLaMo-1B | `encode_query()` | `encode_document()` | 需要 `sentencepiece`；遠端程式碼建立時用 `transformers` 4.47 |
| Sarashina v1 | 不需要 | 不需要 | 非商用 |
| Sarashina v2 | `task: {指令}\nquery: {內容}` | `text: {內容}` | 模型卡指定 `sentence-transformers==4.0.2` |
| e5-small | `query: ` | `passage: ` | — |
| BGE-M3 | 不需要 | 不需要 | 只有 `pytorch_model.bin`，沒有 safetensors |
| EmbeddingGemma | `task: search result \| query: {內容}` | `title: none \| text: {內容}` | 要登入 Hugging Face 並同意條款；不可用 float16 |
| Qwen3-0.6B | `Instruct: {指令}\nQuery:{內容}` | 不需要 | `transformers>=4.51`；模型卡說少了指令，檢索效果約降 1–5% |
| Nemotron-1B | `query: ` | `passage: ` | `transformers>=5.2`、`sentence-transformers>=5.4.1`；模型卡列 Ampere、Blackwell、Hopper，測試硬體是 A100、H100，沒有 Jetson |
| LFM2.5-350M | `query: ` | `document: ` | 自訂模型架構 |

**相依性風險：** 目前最新的 `transformers` 是 5.22.2，Nemotron 需要 5.2 以上，但 PLaMo、RoSEtta 的遠端程式碼是在 4.x 時期寫的。所有模型能否共用同一個環境，要到 5.1.2 實測才知道。

## 公開分數

### 同一組資料集（可以直接比較）

MTEB 官方結果庫的 nDCG@10，只取日文部分。這些模型都有下面 7 個資料集的分數。平均值是本調查自行計算的，不是榜單官方數字。

| 模型 | 7 個資料集平均 | JaGovFaqs | JaCWIR | Mintaka |
|---|---:|---:|---:|---:|
| Sarashina v2（非商用） | 85.54 | 75.61 | 85.52 | 43.65 |
| **ruri-v3-310m** | **83.98** | 77.26 | 84.05 | 33.53 |
| Sarashina v1（非商用） | 83.84 | 72.23 | 82.54 | 40.25 |
| **ruri-v3-130m** | **83.80** | 77.29 | 83.49 | 32.83 |
| ruri-v3-70m | 82.45 | 74.61 | 82.01 | 29.74 |
| **ruri-v3-30m** | **81.94** | 72.36 | 81.93 | 28.36 |
| EmbeddingGemma | 81.28 | 72.09 | 85.24 | 26.83 |
| **Qwen3-0.6B** | **80.30** | 67.82 | 83.11 | 25.32 |
| BGE-M3 | 79.59 | 70.66 | 85.09 | 21.45 |
| **e5-small** | **72.71** | 65.56 | 85.54 | 20.70 |

粗體是建議首輪的模型。7 個資料集是 JaGovFaqs、JaCWIR、Mintaka 和 4 個 NLPJournal。**NLPJournal 是學術論文，分數接近飽和，和對話記憶差距也大**，所以另外列出 3 個較短文本的資料集。逐資料集分數見 [`mteb_japanese_retrieval.json`](mteb_japanese_retrieval.json)。

### 舊版 JMTEB（來自模型卡）

GLuCoSE、RoSEtta、PLaMo 不在 MTEB 結果庫裡，只能看模型卡。這些數字是模型開發者自行公布，和上表基準不同，不能混在一起比較。

| 模型 | Retrieval | 來源 |
|---|---:|---|
| ruri-v3-310m | 81.89 | ruri-v3 模型卡 |
| ruri-v3-130m | 81.89 | ruri-v3 模型卡 |
| ruri-v3-70m | 79.96 | ruri-v3 模型卡 |
| PLaMo-1B | 79.94 | PLaMo、ruri-v3 模型卡 |
| ruri-v3-30m | 78.08 | ruri-v3 模型卡 |
| Sarashina v1 | 77.61 | ruri-v3 模型卡 |
| GLuCoSE v2 | 73.36 | GLuCoSE v2、ruri-v3 模型卡 |
| RoSEtta | 73.21 | RoSEtta 模型卡 |
| e5-small | 67.27 | ruri-v3 模型卡 |

ruri-v3-130m 和 310m 的分數完全相同，兩張模型卡都這樣寫，但其他欄位不同，不確定是否誤植。

### 沒有可比較分數的模型

- **Nemotron-1B：** 結果庫沒有 JMTEB 資料集。`JapaneseLegal1Retrieval` 為 85.40，同資料集 Qwen3-0.6B 是 74.14、BGE-M3 是 70.38，但只有這一個法律資料集，不足以下結論。
- **LFM2.5-350M：** 只有模型卡的 NanoBEIR 日文 nDCG@10 0.575，以及 MKQA 日文 Recall@20 0.685，都是開發者自行公布。

## 篩選理由

| 模型 | 建議 | 理由 |
|---|---|---|
| ruri-v3-30m | 首輪 | 最小（140 MiB），可測品質下限與延遲上限 |
| ruri-v3-130m | 首輪 | 分數接近 310m，權重不到一半 |
| ruri-v3-310m | 首輪 | 可商用模型中，共同資料集分數最高 |
| GLuCoSE v2 | 首輪 | 不同家族的日文模型，增加多樣性 |
| Qwen3-0.6B | 首輪 | 無須申請、可商用的新一代多語模型 |
| e5-small | 首輪 | 輕量且廣泛使用的多語基準 |
| EmbeddingGemma | 備選 | 分數略高於 Qwen3，但要登入並同意條款 |
| Nemotron-1B | 備選 | 日文法律資料集分數亮眼，但約 2.1 GiB，且沒有 JMTEB 分數，Jetson 也未列在支援硬體中 |
| PLaMo-1B | 備選 | 舊版分數比 ruri-v3-130m 低，權重卻是 4 倍，還要執行遠端程式碼 |
| ruri-v3-70m | 不列入 | 介於 30m 和 130m 之間，先不佔名額 |
| RoSEtta | 不列入 | 分數不突出，還要執行遠端程式碼 |
| BGE-M3 | 不列入 | 約 2.1 GiB，分數不如更小的模型，也沒有 safetensors |
| Sarashina v1、v2 | 不列入 | 非商用（D3），約 4.6 GiB 也超過 3 GiB 記憶體門檻 |
| LFM2.5-350M | 不列入 | 授權有營收限制，也沒有可比較的分數 |

**記憶體尚未驗證：** 權重大小不等於執行時的記憶體用量。接近 2 GiB 的模型加上 PyTorch 和向量庫後，可能超過 3 GiB 門檻，要等 5.1.2 量測。

## 其他看過但未列入

| 模型 | 原因 |
|---|---|
| `retrieva-jp/amber-base`、`amber-large` | 舊版 JMTEB Retrieval 為 73.40、75.40，低於同尺寸的 ruri-v3 |
| `jinaai/jina-embeddings-v3` | CC-BY-NC 4.0，非商用 |
| `Alibaba-NLP/gte-multilingual-base`、`Snowflake/snowflake-arctic-embed-m-v2.0`、`ibm-granite/granite-embedding-107m-multilingual`、`nomic-ai/nomic-embed-text-v2-moe` | 結果庫的 JMTEB 資料集分數太少或沒有 |
| `vllm-sr/Vela-1.0-Encoder-307M-Embedding`、`mahiyama/Sazanami-Embed-310m` | 2026 年新模型，結果庫沒有分數，模型卡也沒有細讀 |
| `BAAI/bge-base-zh-v1.5` | 中文模型，不是日文候選；是 `50023` 產品目前使用的模型，可視需要當對照組 |

## 固定版本

5.1.2 下載時使用這些 revision。

| 模型 | revision |
|---|---|
| `cl-nagoya/ruri-v3-30m` | `24899e5de370b56d179604a007c0d727bf144504` |
| `cl-nagoya/ruri-v3-70m` | `07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc` |
| `cl-nagoya/ruri-v3-130m` | `e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6` |
| `cl-nagoya/ruri-v3-310m` | `18b60fb8c2b9df296fb4212bb7d23ef94e579cd3` |
| `pkshatech/GLuCoSE-base-ja-v2` | `a6cbdf4d0b7e1ab4f7ddd85249a4fb96cda54704` |
| `pkshatech/RoSEtta-base-ja` | `9d61cad3e7a6edf946169627ffc2bdca823d39a1` |
| `pfnet/plamo-embedding-1b` | `51507f82e5884f7a9f2306038cfd4104b56748de` |
| `sbintuitions/sarashina-embedding-v1-1b` | `d060fcd8984075071e7fad81baff035cbb3b6c7e` |
| `sbintuitions/sarashina-embedding-v2-1b` | `1f3408afaa7b617e3445d891310a9c26dd0c68a5` |
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `google/embeddinggemma-300m` | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `nvidia/Nemotron-3-Embed-1B-BF16` | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| `LiquidAI/LFM2.5-Embedding-350M` | `f35ae2c91d687658dbf1f2b449382f0b019b9808` |

e5-small、Qwen3、Nemotron 在結果庫的分數可能來自較舊的 revision，上表是目前最新的 revision。

## 調查限制

- **榜單讀不到：** MTEB Leaderboard 是動態網頁，抓不到內容，所以改讀它的原始結果庫。
- **自算平均：** 平均值由本調查計算；Sarashina v2 模型卡的 Retrieval 是 76.48，用 11 個資料集自算是 76.63，略有差異。
- **模型卡內容：** 規格和舊版分數來自模型卡，由開發者自行公布；部分欄位是透過網頁摘要工具讀取，再用設定檔或另一張模型卡交叉核對。
- **未採用的資料：** ANLP 2026 論文 P1-4 的 PDF，日文文字抽出來是亂碼，只讀得到表格數字，所以沒有當作依據。

## 來源

- [MTEB 結果庫](https://github.com/embeddings-benchmark/results)（commit `922b3df446f88756d954e4099b7211bcafaf99c9`）
- [JMTEB](https://github.com/sbintuitions/JMTEB)、[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- 模型卡：[ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)、[ruri-v3-130m](https://huggingface.co/cl-nagoya/ruri-v3-130m)、[ruri-v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-30m)、[GLuCoSE v2](https://huggingface.co/pkshatech/GLuCoSE-base-ja-v2)、[RoSEtta](https://huggingface.co/pkshatech/RoSEtta-base-ja)、[PLaMo-1B](https://huggingface.co/pfnet/plamo-embedding-1b)、[Sarashina v2](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b)、[e5-small](https://huggingface.co/intfloat/multilingual-e5-small)、[BGE-M3](https://huggingface.co/BAAI/bge-m3)、[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)、[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[Nemotron-3-Embed-1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)、[LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M)
- 規格與檔案大小：Hugging Face API `https://huggingface.co/api/models/<模型>?blobs=true`、各模型的 `config.json`
