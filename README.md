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

### 環境變數

| 變數 | 說明 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 金鑰（僅 `--llm` 模式需要） |

Windows 建議使用 `python -X utf8 main.py` 避免中文亂碼。

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
│   └── report_writer.py      # PDF 報告產生（reportlab）
├── tech/
│   └── technical_spec.md     # 各指標計算方式技術文件
└── reports/                  # 執行後自動產生（不納入版控）
```

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

Tripwire 觸發時，綜合分數強制拉高至 ≥ 0.95。

## 技術文件

詳細指標計算公式、設計決策與參數調整指南請見 [`tech/technical_spec.md`](tech/technical_spec.md)。

## 開發環境

- Python 3.11+
- reportlab（PDF 產生）
- anthropic SDK（I3 LLM 抽取，選用）
