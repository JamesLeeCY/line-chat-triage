# LINE 群組健康度分診系統

自動分析 LINE 工作群組對話，計算多維度健康指標，產出 PDF 優先處理報告。

## 功能概覽

- **I1** 未回應提問年齡（業務時間感知）
- **I2** 客戶回應延遲 P90（業務時間計算）
- **I3** 最老未解議題年齡（Claude LLM 抽取，選用）
- **I4** 客戶負面情緒比例（72h 視窗、情緒狀態標籤）
- **I5** 近 24h 訊息量
- **I6** 語義熵時間序列（Semantic Entropy，對話主題複雜度追蹤）
- **Tripwire** 非補償性升級偵測（退款 / 投訴 / 找主管等關鍵詞）
- 自動產出 PDF 總覽報告（群組排名 + 各群組逐項分析 + I6 sparkline 圖表）

---

## 範例報告

以下為 10 個工作群組的實際分析輸出（使用去識別化 pseudo 對話）。

📄 [下載完整 PDF 報告](docs/sample_report/sample_report.pdf)

### 第 1 頁 — 群組總覽與排名

![總覽頁](docs/sample_report/page_01.png)

### 第 2 頁 — 最高風險群組詳細分析

![住宅翻新_陳太太](docs/sample_report/page_02.png)

### 第 3 頁 — I6 語義熵 sparkline 範例（健身房規劃群組）

![健身房規劃](docs/sample_report/page_03.png)

### 第 4–5 頁 — 中風險群組分析

![科技公司大廳](docs/sample_report/page_04.png)

![品牌旗艦店](docs/sample_report/page_05.png)

<details>
<summary>📋 查看剩餘 9 頁（點擊展開）</summary>

![page 6](docs/sample_report/page_06.png)
![page 7](docs/sample_report/page_07.png)
![page 8](docs/sample_report/page_08.png)
![page 9](docs/sample_report/page_09.png)
![page 10](docs/sample_report/page_10.png)
![page 11](docs/sample_report/page_11.png)
![page 12](docs/sample_report/page_12.png)
![page 13](docs/sample_report/page_13.png)
![page 14](docs/sample_report/page_14.png)

</details>

---

## 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 準備資料

1. 將 LINE 匯出的對話 `.txt` 檔放入 `data/conversations/`
2. 編輯 `data/employees.txt`，每行填入一位員工顯示名稱

`employees.txt` 格式：
```
成員A
成員B
成員C
```

### 執行

```bash
# 基本執行（以目前時間為基準）
python main.py

# 指定基準時間（模擬/回測）
python main.py --now 2025-09-05T17:00

# 產出 PDF 報告
python main.py --now 2025-09-05T17:00 --report

# 啟用 LLM 議題抽取（需 ANTHROPIC_API_KEY）
python main.py --report --llm --llm-threshold 0.3

# 指定對話資料夾與報告輸出目錄
python main.py --conversations data/conversations --report-dir reports
```

### 測試

```bash
pip install pytest httpx
python -m pytest
```

### 匯入 Telegram 群組歷史

1. 電腦版 Telegram Desktop →「設定」→「進階」→「匯出 Telegram 資料」
2. 只勾選需要的群組類型（私人群組／公開群組），格式選 **JSON**，媒體檔可全部取消勾選
3. 將匯出資料夾放到 `data/telegram/`（已排除於版控），列出成員 id 以填寫 `employees.txt`：
   ```bash
   python -X utf8 -m src.telegram_export data/telegram
   ```
   `employees.txt` 可填顯示名稱，或 Telegram id（如 `user123456789`，改名也不受影響）
4. 執行 triage：
   ```bash
   python -X utf8 main.py --telegram-export data/telegram --report
   ```

只處理群組；私訊、頻道、Bot 對話與收藏訊息會自動略過。

### 社群模式（股票社群群組：多空、話題、entropy）

客服 triage 以外的第二種模式，針對大型投資社群群組。流程與資料都在 `data/community/`（已排除於版控）。
目前進度、實驗結果與下一步記錄在 [HANDOFF.md](HANDOFF.md)。

```bash
python -X utf8 -m src.community prepare        # Telegram 匯出 → SQLite（只需一次）
python -X utf8 -m src.community sample         # 分層抽樣 3,000 則，500 則保留為考卷
python -X utf8 -m src.community label --limit 20             # 地端 Qwen3 標註（Ollama，可中斷續跑）
python -X utf8 -m src.community label --backend claude       # 或改用 Claude Haiku（需 ANTHROPIC_API_KEY）
python -X utf8 -m src.community review-queue   # 待標清單：AI 判多空全收＋中立對照抽樣
python -X utf8 -m src.community annotate --queue  # 本機人工標註網頁 http://127.0.0.1:8770
python -X utf8 -m src.community compare        # 各模型標註 vs 人工標註（分層加權，accuracy / macro-F1 / kappa）
python -X utf8 -m src.community train --labels qwen3-8b --test-labels human --show-features
```

標註來源存成 `data/community/labels/<名稱>.jsonl`：模型以模型名命名（如 `qwen3-8b`），人工標註為 `human`。
分類器（`src/community/classifier.py`）介面為 `fit` / `predict_proba`，可替換為其他模型（如 Jev）在同一份考卷上比較。

#### 目前結果（2026-10-06，人工標註 165 則完成）

資料：2 個 Telegram 股票社群，共約 109 萬則訊息。人工只標 AI 有爭議的訊息：AI 判有多空的**全部收錄**，AI 判中立的**抽樣作對照組**，
再依分層權重（該層總數 ÷ 已標數）推估整份考卷的分數。隨機考卷（`test`，500 則）與加強考卷（`enrich_test`，200 則，只抽含多空/部位/代號用語的訊息）分開報告。
本 repo 公開，以下只列統計數字，不含任何訊息內容。

**人工標註的分布**

| | 隨機考卷 | 加強考卷 |
|---|---|---|
| AI 判多空的訊息，人工也判有多空 | 23 / 43 | 22 / 48 |
| AI 判中立的對照抽樣，人工判有多空 | 3 / 59 | 0 / 15 |
| 推估有多空立場的比例 | 約 9%（看空約 25、看多約 21 則） | 約 11% |

**LLM 標註器 vs 人工（多空，加權推估）**

| 標註來源 | 考卷 | accuracy | macro-F1 | Cohen's kappa |
|---|---|---|---|---|
| Qwen3 8B 提示詞 v1 | 隨機 | 0.906 | **0.600** | **0.415** |
| Qwen3 8B 提示詞 v2 | 隨機 | 0.912 | 0.561 | 0.394 |
| Qwen3 8B 提示詞 v2 | 加強 | 0.860 | 0.688 | 0.549 |

- **主要錯誤是「誤判有多空」**：AI 判有多空的訊息，人工同意方向的只有約 45%（v1 18/40、v2 14/31、v2 加強 20/48），其餘多半被人工判為中立，少數方向相反。
- **v1 與 v2 大致打平**：v2 減少誤判（AI 多空中被人工判中立：v1 19 則 → v2 12 則），但也漏掉真正的多空，看空召回率約 27% → 20%、看多約 53% → 43%。
  v1 與 v2 意見不同的 15 則中，v2 對 8 則、v1 對 5 則；v2 改成中立的 12 則裡，有 4 則人工判定確實有多空，證實 v2 有矯枉過正。
- 話題分類一致性低（macro-F1 0.22–0.24）：AI 常把閒聊標成個股、新聞或操作部位。
- 隨機考卷中，AI 判中立的 457 則只抽 59 則，其中 3 則人工判有多空，每則權重約 7.7，因此隨機考卷的召回率估計**誤差範圍很大**。

**分類器 vs 人工（多空，加權推估）**：TF-IDF（字元 n-gram）+ 邏輯迴歸，訓練資料為 v2 標註的 `enrich_train` 800 則

| 模型 | 隨機考卷 macro-F1 | 加強考卷 macro-F1 |
|---|---|---|
| 永遠猜中立（基準） | 0.32 | 0.31 |
| TF-IDF + 邏輯迴歸 | 0.41 | 0.65 |

- 加強考卷上看多 / 看空的 F1 約 0.52 / 0.47；隨機考卷上只有 0.22 / 0.07，在真實分布下仍不可用。
- 先前以 v2 標註當答案得到 0.49 / 0.50；改用人工答案後，隨機考卷分數下降，代表訓練標註本身的雜訊是主要瓶頸。

**下一步**：提示詞 v3 的目標是保留 v1 的召回率，同時壓低誤判多空。定稿後再標註第二輪加強抽樣（2,000 則），重新訓練並以人工答案評分。
細節見 [HANDOFF.md](HANDOFF.md)。

### 接收 LINE 即時訊息（webhook）

LINE Bot 只能收到它加入群組**之後**的訊息；加入前的歷史仍需用匯出 `.txt`。

1. 在 LINE Official Account Manager 建立官方帳號並啟用 Messaging API，於 LINE Developers Console 取得 **Channel secret** 與 **Channel access token**
2. 官方帳號設定：開啟「允許加入群組」，並**關閉自動回應訊息與加入好友歡迎訊息**，避免 Bot 在客戶群組裡發言
3. 啟動 webhook（開發時可用 `ngrok http 8000` 取得公開 HTTPS 網址）：
   ```bash
   uvicorn src.line_webhook:create_app_from_env --factory --port 8000
   ```
4. Developers Console → Webhook URL 填 `https://<host>/callback`，開啟 Use webhook，按 Verify
5. 把官方帳號邀進群組；收到訊息後列出成員 userId，將員工的 userId 加入 `employees.txt`（可加 `# 註解` 標示姓名）：
   ```bash
   python -X utf8 -m src.line_store data/line.db
   ```
6. 從資料庫跑 triage：
   ```bash
   python -X utf8 main.py --line-db data/line.db --report
   ```

不接 LINE 也能在本機演練整條路徑：啟動 webhook 後，用匯出檔重播成已簽章的事件：

```bash
python -X utf8 tools/replay_export.py data/conversations/*.txt --url http://localhost:8000/callback
```

### 環境變數

| 變數 | 說明 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 金鑰（僅 `--llm` 模式需要） |
| `LINE_CHANNEL_SECRET` | 驗證 webhook 簽章（webhook 必填） |
| `LINE_CHANNEL_ACCESS_TOKEN` | 查詢群組名稱與成員顯示名稱（選填；未設定時以 userId 顯示） |
| `LINE_DB_PATH` | webhook 資料庫路徑（預設 `data/line.db`） |

Windows 建議使用 `python -X utf8 main.py` 避免中文亂碼。

---

## 專案結構

```
line_chat/
├── main.py                   # 主程式入口
├── requirements.txt
├── data/
│   ├── employees.txt         # 員工名單
│   └── conversations/        # LINE 匯出對話 .txt
├── src/
│   ├── parser.py             # LINE 匯出格式解析
│   ├── enrichment.py         # 對話行為分類、情緒評分、升級偵測
│   ├── metrics.py            # I1–I6 指標計算
│   ├── dashboard.py          # 終端機排名輸出
│   ├── llm_extractor.py      # I3 LLM 議題抽取（Claude）
│   ├── report_writer.py      # PDF 報告產生（reportlab）
│   ├── line_webhook.py       # LINE webhook 接收（FastAPI、簽章驗證、名稱查詢）
│   ├── line_adapter.py       # webhook 事件 → 資料列
│   ├── line_store.py         # SQLite 儲存，輸出與 parse_file 相同的 Message
│   ├── telegram_export.py    # Telegram Desktop JSON 匯出解析
│   └── community/            # 社群模式：SQLite 儲存、抽樣、標註器、人工標註網頁、分類器、評估
├── tools/
│   └── replay_export.py      # 將匯出檔重播為 webhook 事件（本機演練用）
├── tests/                    # pytest
├── tech/
│   └── technical_spec.md     # 各指標計算方式技術文件
├── docs/
│   └── sample_report/        # 範例報告截圖與 PDF
└── reports/                  # 執行後自動產生（不納入版控）
```

---

## 指標說明

| 指標 | 意義 | 警示閾值 | 臨界閾值 |
|------|------|---------|---------|
| I1 | 最老未回覆客戶提問的業務時間 | 30 分 | 360 分 |
| I2 | 回應延遲 P90（業務時間） | 120 分 | 480 分 |
| I3 | 最老未解議題年齡（牆鐘時間） | 60 分 | 480 分 |
| I4 | 72h 內客戶訊息負面比例 | 10% | 60% |
| I5 | 近 24h 訊息總量（參考用） | — | — |
| I6 | Bigram Shannon 熵時序（bits） | 診斷用，不計入綜合分 | — |

**綜合分數** = 0.35×I1 + 0.20×I2 + 0.15×I3 + 0.30×I4

Tripwire 觸發時，綜合分數強制拉高至 ≥ 0.95。升級詞只計客戶訊息，且只看最近 72 小時。

---

## 技術文件

詳細指標計算公式、設計決策與參數調整指南請見 [`tech/technical_spec.md`](tech/technical_spec.md)。

## 開發環境

- Python 3.11+
- reportlab（PDF 產生）
- anthropic SDK（I3 LLM 抽取，選用）
