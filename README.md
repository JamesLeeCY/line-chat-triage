# LINE 群組健康度 Triage 系統

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
│   └── line_store.py         # SQLite 儲存，輸出與 parse_file 相同的 Message
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
