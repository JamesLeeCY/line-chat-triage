# LINE 群組健康度 Triage 系統 — 技術規格文件

**版本**：1.1.0  
**更新日期**：2026-06-19  
**對應原始碼**：`src/parser.py`、`src/enrichment.py`、`src/metrics.py`、`src/llm_extractor.py`

---

## 目錄

1. [系統概觀](#1-系統概觀)
2. [資料輸入格式](#2-資料輸入格式)
3. [訊息前處理](#3-訊息前處理)
4. [對話行為分類](#4-對話行為分類)
5. [情緒評分](#5-情緒評分)
6. [升級標記偵測](#6-升級標記偵測)
7. [指標計算](#7-指標計算)
   - [I1 未回應提問年齡](#i1-未回應提問年齡-unanswered-question-age)
   - [I2 客戶回應延遲 P90](#i2-客戶回應延遲-p90-response-latency-p90)
   - [I3 最老未解議題年齡（LLM）](#i3-最老未解議題年齡-oldest-open-issue-age)
   - [I4 客戶負面情緒](#i4-客戶負面情緒-customer-negative-sentiment)
   - [I5 近期訊息量](#i5-近期訊息量-volume)
8. [Tripwire 紅旗機制](#8-tripwire-紅旗機制)
9. [綜合分數（Composite Score）](#9-綜合分數-composite-score)
10. [群組排名邏輯](#10-群組排名邏輯)
11. [業務時間定義](#11-業務時間定義)
12. [LLM 議題抽取（Phase 2）](#12-llm-議題抽取-phase-2)
13. [設計決策與限制](#13-設計決策與限制)
14. [參數調整指南](#14-參數調整指南)

---

## 1. 系統概觀

本系統針對多個 LINE 工作群組進行健康度評估，目標是協助主管快速識別**哪些群組需要立即介入**，而非逐一閱讀每個群組的聊天記錄。

### 設計原則

- **Precision@K 優先**：寧可多報一個需要關注的群組，也不要漏掉真正有問題的群組。
- **非補償性升級（Non-compensatory Escalation）**：Tripwire 一旦觸發，無論其他指標多低，群組都會排到最前面。單一嚴重訊號不能被其他良好指標「平均掉」。
- **分層執行（Tiered Execution）**：便宜的規則型指標（I1、I2、I4）對所有群組持續計算；昂貴的 LLM 指標（I3）只在規則層已發出警示時才啟動。
- **業務時間感知**：所有時間類指標僅計算業務時間內的分鐘數，避免週末或深夜空窗期誇大延遲數字。

### 系統架構

```
LINE 匯出 .txt
       │
       ▼
  [parser.py]          解析 → Message 物件列表
       │
       ▼
  [enrichment.py]      對話行為分類、情緒評分、升級標記
       │
       ▼
  [metrics.py]         計算 I1 / I2 / I4 / I5 / Tripwire / Composite
       │
       ├──(觸發條件)──▶ [llm_extractor.py]  I3 LLM 議題抽取（選用）
       │
       ▼
  [dashboard.py]       排名 + 終端機輸出
       │
       ▼
  [report_writer.py]   PDF 報告產生
```

---

## 2. 資料輸入格式

### LINE 匯出格式規範

```
[LINE] 群組名稱
儲存日期：YYYY/MM/DD HH:MM

YYYY.MM.DD 星期X
HH:MM\t發送人\t訊息內容
HH:MM\t發送人\t貼圖
```

- 日期分隔行（`YYYY.MM.DD 星期X`）出現時更新當前日期上下文。
- 訊息行由 Tab（`\t`）分隔三欄：時間、發送人、內容。
- 特殊內容識別：`貼圖` / `[貼圖]` / `Sticker` 視為貼圖；`圖片` / `[圖片]` / `Photo` / `Image` 視為圖片。

### 員工識別

員工帳號清單存放於 `data/employees.txt`，每行一個帳號名稱（以 `#` 開頭的行視為註解）。

```python
def load_employees(path: str) -> set[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}
```

解析每則訊息時，`sender in employees` 為 True 則 `role = "staff"`，否則 `role = "customer"`。

### Message 資料結構

| 欄位 | 型別 | 說明 |
|------|------|------|
| `group_id` | str | 群組名稱（來自第一行或檔名） |
| `msg_id` | str | `{group_id}_{序號:05d}` |
| `sender` | str | 原始發送人名稱 |
| `role` | str | `staff` \| `customer` \| `system` |
| `timestamp` | datetime | 訊息時間戳（無時區） |
| `text` | str | 訊息文字（貼圖/圖片為空字串） |
| `is_sticker` | bool | 是否為貼圖 |
| `is_image` | bool | 是否為圖片 |
| `dialogue_act` | str | 對話行為（enrichment 後填入） |
| `sentiment` | float | 情緒分數 [-1, 1]（enrichment 後填入） |
| `is_escalation_marker` | bool | 是否含升級詞彙（enrichment 後填入） |

---

## 3. 訊息前處理

### 訊息碎片合併（Fragment Merging）

同一發送人在 60 秒以內連續發送的多則訊息，合併為一則邏輯訊息。

**合併條件**（三項需同時成立）：
1. `msg.sender == prev.sender`（同一發送人）
2. `(msg.timestamp - prev.timestamp).total_seconds() <= 60`（時間差 ≤ 60 秒）
3. 前後訊息皆非貼圖或圖片

**合併方式**：文字以空格串接，`msg_id` 保留第一則的 ID，`timestamp` 保留第一則的時間。

**設計理由**：LINE 使用者習慣把一段話拆成多則快速傳送（如「好」「謝謝」「那我們週三見」），若不合併會誇大訊息數量，也會讓對話行為分類失準。

---

## 4. 對話行為分類

**原始碼**：`src/enrichment.py :: classify_dialogue_act()`

每則訊息依序對照以下規則，命中第一條規則即停止：

| 優先序 | 對話行為 | 觸發條件 |
|--------|----------|----------|
| 1 | `ack` | 貼圖 / 圖片；或符合確認句 pattern（見下） |
| 2 | `question` | 訊息含疑問語氣（見下） |
| 3 | `request` | 訊息含請求語氣（見下） |
| 4 | `report` | 訊息含抱怨描述（見下） |
| 5 | `chit_chat` | 閒聊語氣 |
| 6 | `answer` | 其餘（預設值） |

**注意**：`ack` 的判斷優先於 `question` 和 `request`，避免「好，麻煩你們了！」這類確認語被誤判為 request 導致計入 I2 延遲。

### 各行為的 Pattern 清單

**ACK 確認句**
```
^(好|ok|OK|收到|了解|知道了|沒問題|沒事|好的|好哦|嗯|恩|是|是的|收|ㄟ好)[!！。\s]*$
^(感謝|謝謝|感恩|謝)[!！。\s]*$
^好[，,]?(麻煩|辛苦|感謝|謝).{0,15}$
👍（含此 emoji）
```

**QUESTION 問題**
```
[嗎呢？?]$          # 以疑問助詞或問號結尾
^是否               # 是否...
^可以.{0,10}[嗎？?] # 可以...嗎？
^有沒有             # 有沒有...
^請問               # 請問...
^能不能             # 能不能...
可以嗎 / 方便嗎 / 好嗎 / 對嗎 / 這樣可以
```

**REQUEST 請求**
```
可以.{0,15}(給|提供|傳|發|做|改|幫)
(給|幫|請).{0,10}(我|一下)
(需要|想要|希望).{0,15}(可以|能)
麻煩 / 勞駕
```

**REPORT 抱怨回報**
```
(太小|太大|不夠|不對|不好|不滿|有問題)
(覺得|感覺).{0,10}(不|沒)
```

**CHIT_CHAT 閒聊**
```
^(早安|晚安|哈哈|😂|🥹|🙌|貼圖)[!！。\s]*$
颱風（含此字即為閒聊）
```

### 分類對指標的影響

- `question` / `request` 角色為 customer → 計入 **I1**（未回覆偵測）和 **I2**（延遲計算）
- 其餘對話行為不影響 I1 / I2

---

## 5. 情緒評分

**原始碼**：`src/enrichment.py :: score_sentiment()`

使用詞典比對法，計算訊息的情緒分數。

### 公式

```
score = (正面詞命中數 - 負面詞命中數) / (正面詞命中數 + 負面詞命中數)
```

若無任何詞命中則回傳 0.0。分數範圍為 `[-1.0, 1.0]`，負值代表負面。

### 詞典

**負面詞**（22 個）：
`哭哭、拍謝、抱歉、對不起、不好、太小、太貴、不夠、不滿、有問題、困擾、麻煩、擔心、疑慮、可惜、遺憾、失望、退費、投訴、換人、不合理`

**正面詞**（15 個）：
`感謝、謝謝、感恩、好的、沒問題、完美、很好、讚、棒、期待、開心、順利、👍、🙌、🥹`

### 設計限制

- 詞典比對不考慮語境，「麻煩你們了」中的「麻煩」會計入負面詞（即使語意是禮貌請求）。
- 否定句（「沒有問題」）可能因「問題」被算負面而誤判，但「沒問題」整體會被正面詞命中修正。
- Phase 3 計畫改用語意模型替換此方法。

---

## 6. 升級標記偵測

**原始碼**：`src/enrichment.py :: has_escalation_marker()`

逐一檢查訊息文字是否包含以下任何關鍵詞（子字串比對，不區分大小寫邊界）：

| 類別 | 關鍵詞 |
|------|--------|
| 退款要求 | 退費、退錢、退款、退訂金、退訂 |
| 投訴行為 | 投訴、換人 |
| 升級對象 | 找你主管、找你老闆、找主管、找老闆、跟你主管談、跟你老闆談 |
| 法律行動 | 解約、取消合約、告你、法院、消保、消費者保護 |

命中任一關鍵詞，該訊息的 `is_escalation_marker` 設為 `True`。

**重要**：升級標記偵測針對**所有歷史訊息**（不限 24 小時），只要群組對話中曾出現過升級詞，永久保持警示狀態。

---

## 7. 指標計算

所有指標的 severity 值範圍為 `[0.0, 1.0]`，透過線性插值函數 `_age_to_severity()` 轉換：

```python
def _age_to_severity(minutes: float, warn_min: float, crit_min: float) -> float:
    if minutes < warn_min:  return 0.0
    if minutes >= crit_min: return 1.0
    return (minutes - warn_min) / (crit_min - warn_min)
```

- `minutes < warn_min`：無風險，severity = 0
- `warn_min ≤ minutes < crit_min`：線性爬升
- `minutes ≥ crit_min`：最高風險，severity = 1

---

### I1 未回應提問年齡（Unanswered Question Age）

**代表意義**：群組中最老的一個「客戶提問尚無任何員工回覆」的年齡。  
用來偵測**被遺漏的提問**，即員工沒有注意到或刻意忽略的問題。

**計算流程**：

1. 篩選所有 `role == "customer"` 且 `dialogue_act in ("question", "request")` 的訊息
2. 對每則符合的訊息，向後掃描是否有 `role == "staff"` 的訊息存在
3. 若無員工回覆（`responded == False`），計算該訊息到 `now` 的業務時間分鐘數
4. 取所有未回覆提問中 severity 最高者作為 `i1_severity`

**Severity 轉換參數**：

| 閾值 | 業務分鐘數 | 說明 |
|------|------------|------|
| warn_min | 30 分 | 超過 30 分鐘開始計入風險 |
| crit_min | 360 分（6h） | 超過 6 業務小時達到最高風險 |

**Tripwire 條件**：若 `i1_oldest_age_min ≥ 480`（8 業務小時），強制觸發 Tripwire（見第 8 節）。

**輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `i1_severity` | 最高風險提問的 severity [0,1] |
| `i1_open_questions` | 目前開放（未回覆）提問總數 |
| `i1_oldest_age_min` | 最老未回覆提問的業務時間年齡（分鐘） |

---

### I2 客戶回應延遲 P90（Response Latency P90）

**代表意義**：在所有客戶提問中，第 90 百分位數的回應時間。  
P90 = 10 個問題中最慢的那個（而不是最快的），能合理反映「大多數情況下員工多快能回應」。

**計算流程**：

1. 篩選 `role == "customer"` 且 `dialogue_act in ("question", "request")` 的訊息
2. 對每則提問，找第一則在其之後出現的 `role == "staff"` 訊息
3. 若兩則訊息的日曆時間差 > 3 天，視為同一議題在不同時段延續，排除此配對（避免跨日空窗期誇大延遲）
4. 計算配對的業務時間分鐘數，加入 latency 清單
5. 排序後取第 90 百分位（`index = int(n * 0.9)`, 若越界則取最後一個）

**Severity 轉換參數**：

| 閾值 | 業務分鐘數 | 說明 |
|------|------------|------|
| warn_min | 120 分（2h） | P90 超過 2 業務小時開始計風險 |
| crit_min | 480 分（8h） | P90 超過 8 業務小時達到最高風險 |

**與 I1 的差異**：

| | I1 | I2 |
|--|----|----|
| 衡量對象 | 目前**仍未回覆**的問題 | **所有**問題的回應速度分佈 |
| 時間範圍 | 全部歷史（含現在） | 有回覆的歷史問答配對 |
| 風險類型 | 「漏接」風險 | 「慢回」風險 |

**輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `i2_severity` | P90 對應的 severity [0,1] |
| `i2_p90_min` | P90 原始業務時間分鐘數 |

---

### I3 最老未解議題年齡（Oldest Open Issue Age）

**代表意義**：透過 LLM 語意理解識別的「開放議題」中，從提出到現在最久的一個年齡。  
比 I1 更語意化：I1 只看訊息有沒有被員工「回覆過」，I3 看議題有沒有被「實質解決」。

**觸發條件（Tiered Execution）**：只有當以下任一條件成立時才呼叫 LLM：
- `m.tripwire == True`
- `m.composite >= llm_threshold`（預設 0.3）

**計算流程（`compute_i3()`）**：

1. 接收 LLM 回傳的 `issues` 清單（見第 12 節）
2. 篩選 `status in ("unresolved", "unclear")` 的議題
3. 對每個開放議題，計算 `raised_at` 到 `now` 的總分鐘數（**不限業務時間**，I3 使用絕對時間）
4. 取最大值作為 `i3_oldest_age_min`

**Severity 轉換參數**：

| 閾值 | 分鐘數 | 說明 |
|------|--------|------|
| warn_min | 60 分（1h） | 開放議題超過 1 小時開始計入 |
| crit_min | 480 分（8h） | 超過 8 小時達到最高風險 |

**注意**：I3 使用**掛鐘時間**（wall clock），非業務時間，因為未解的客訴在業務時間外仍持續累積壓力。

**輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `i3_severity` | 最老未解議題的 severity [0,1] |
| `i3_oldest_age_min` | 最老未解議題年齡（掛鐘分鐘） |
| `i3_open_issues` | 開放議題數量 |
| `i3_issues` | LLM 回傳的原始議題清單（list[dict]） |

---

### I4 客戶負面情緒（Customer Negative Sentiment）

**代表意義**：近 24 小時內，客戶訊息中負面情緒訊息的佔比。  
追蹤的是**趨勢**（近期），而非全部歷史，因此能反映最新的客戶心情變化。

**計算流程**：

1. 篩選 `role == "customer"` 且 `timestamp >= now - 24h` 且 `not is_sticker` 的訊息
2. 統計其中 `sentiment < -0.1` 的訊息數量
3. `i4_neg_ratio = 負面訊息數 / 客戶訊息總數`
4. Severity 轉換：`(neg_ratio - 0.2) / 0.4`，clamp 至 [0, 1]

**Severity 轉換參數**：

| neg_ratio | severity | 說明 |
|-----------|----------|------|
| < 20% | 0.0 | 正常波動範圍，不計入風險 |
| 20% ~ 60% | 0.0 ~ 1.0 | 線性爬升（每增加 1% 提升 0.025） |
| ≥ 60% | 1.0 | 最高風險 |

**情緒閾值**：`sentiment < -0.1` 而非 `< 0`，保留中性緩衝區，避免「麻煩問一下」這類輕微負面詞彙誤觸發。

**輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `i4_severity` | 負面情緒 severity [0,1] |
| `i4_neg_ratio` | 負面訊息比例 [0,1] |

---

### I5 近期訊息量（Volume）

**代表意義**：近 24 小時內的訊息總則數（所有角色）。  
目前作為觀察指標，不直接計入 composite score。高訊息量可能代表議題密集或關係活躍。

**計算流程**：

```python
cutoff_24h = now - timedelta(hours=24)
i5_msg_count_24h = sum(1 for msg in messages if msg.timestamp >= cutoff_24h)
```

---

## 8. Tripwire 紅旗機制

**核心設計**：Tripwire 是一種**非補償性（Non-compensatory）**升級機制。一旦觸發，無論其他指標分數多低，該群組都會：
1. 強制排在所有未觸發 Tripwire 群組的前面
2. `composite` 強制設為 `max(原始 composite, 0.95)`

**觸發條件**（任一成立即觸發）：

| 條件 | 說明 |
|------|------|
| 任一訊息含升級詞彙 | `msg.is_escalation_marker == True`（全部歷史） |
| I1 最老未回覆提問 ≥ 8 業務小時 | `i1_oldest_age_min >= 480` |

**設計理由**：客戶說出「退款」「投訴」「找主管」時，這是一個**方向性訊號**，代表關係已惡化到特定程度。此類訊號的風險不能用「其他指標都很好」來抵銷，因此使用不可補償的紅旗機制確保必定被看到。

**輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `tripwire` | bool，是否觸發 |
| `tripwire_reasons` | list[str]，觸發原因清單（含 msg_id 與訊息摘要） |

---

## 9. 綜合分數（Composite Score）

**代表意義**：將多個指標加權合併為單一數字，用於非 Tripwire 群組的相對排序。

### 公式

**Phase 1（無 LLM）**：
```
composite = 0.35 × I1_severity + 0.20 × I2_severity + 0.30 × I4_severity
```

**Phase 2（含 LLM，I3 已計算）**：
```
composite = 0.35 × I1_severity + 0.20 × I2_severity
           + 0.15 × I3_severity + 0.30 × I4_severity
```

**Tripwire 覆蓋**：
```python
if tripwire:
    composite = max(composite, 0.95)
```

### 權重設定依據

| 指標 | 權重 | 設定理由 |
|------|------|----------|
| I1 未回應提問 | 0.35 | 最直接的服務失效訊號，優先級最高 |
| I4 客戶負面情緒 | 0.30 | 反映關係健康度，短期趨勢敏感 |
| I2 回應延遲 P90 | 0.20 | 結構性效率問題，長期指標 |
| I3 最老未解議題 | 0.15 | LLM 精確度高但執行成本高，權重保守 |

### 分數區間參考

| composite 範圍 | 建議行動 |
|----------------|----------|
| ≥ 0.95（Tripwire） | 立即介入（1 小時內） |
| 0.15 ~ 0.94 | 今日跟進 |
| < 0.15 | 維持現狀 |

---

## 10. 群組排名邏輯

**原始碼**：`src/dashboard.py :: rank_groups()`

```python
tripwire = sorted([m for m in metrics if m.tripwire], key=lambda x: -x.composite)
normal   = sorted([m for m in metrics if not m.tripwire], key=lambda x: -x.composite)
return tripwire + normal
```

1. Tripwire 群組永遠排在所有正常群組前面
2. Tripwire 群組內部依 composite 降序排列
3. 正常群組依 composite 降序排列

---

## 11. 業務時間定義

**原始碼**：`src/metrics.py :: _service_minutes()`

| 參數 | 值 |
|------|-----|
| 業務日 | 週一至週五（weekday 0–4） |
| 業務時段 | 09:00–18:00 |
| 業務時間/天 | 540 分鐘（9 小時） |

### 計算方法

逐分鐘迭代，每個「在業務時間內的分鐘」計 1 分：

```python
def _service_minutes(start: datetime, end: datetime) -> float:
    total = 0.0
    cur = start
    while cur < end:
        if _in_service(cur):
            total += 1
        cur += timedelta(minutes=1)
    return total
```

**效能注意**：此方法對短時間段（< 1 天）效能可接受，但對跨多日的長時間段可能較慢。目前已加入 3 日上限保護（I2 計算），I1 的計算對象為「目前仍開放的問題」，通常不超過數日。

### 為何使用業務時間

深夜 23:00 客戶傳訊，員工隔天 09:30 回覆，掛鐘時間差約 630 分鐘，但業務時間差僅 30 分鐘。若使用掛鐘時間，此案例會觸發 I2 警示，但實際上員工已在上班後立即回覆，不應計為延遲。

---

## 12. LLM 議題抽取（Phase 2）

**原始碼**：`src/llm_extractor.py`

### 使用模型

預設使用 `claude-haiku-4-5`（成本考量）。可於 `extract_issues()` 傳入 `model` 參數覆蓋。

### 議題資料結構

LLM 回傳的每個議題包含以下欄位：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `issue_id` | str | 短流水號（如 ISS-001） |
| `group_id` | str | 系統附加，非 LLM 產生 |
| `raised_by` | str | `customer` \| `staff` |
| `raised_at` | str | ISO8601 UTC 時間戳 |
| `summary` | str | 不超過 40 字的一句話摘要 |
| `type` | str | `question` \| `request` \| `complaint` \| `report` |
| `status` | str | `resolved` \| `unresolved` \| `unclear` |
| `resolution_evidence` | str | 已解決時的依據摘要 |
| `last_activity_at` | str | ISO8601 UTC 時間戳 |
| `evidence_msg_ids` | list[str] | 相關訊息的 msg_id 列表 |

### 判斷準則（System Prompt 節錄）

- 同一客戶短時間內連續追問同一件事，視為一個議題
- staff 有明確回答或確認完成即算 `resolved`
- 判斷不清楚時填 `unclear`，不猜測
- 整段對話無議題時回傳 `{"issues": []}`

### 執行環境

需設定環境變數 `ANTHROPIC_API_KEY`：

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python -X utf8 main.py --llm --llm-threshold 0.3
```

---

## 13. 設計決策與限制

### 已知限制

| 限制 | 影響 | 計畫改善 |
|------|------|----------|
| 情緒詞典為固定規則 | 無法處理語境、否定句 | Phase 3：embedding 語意模型 |
| 對話行為分類為 regex | 複雜句型可能誤判 | 積累真實標注後訓練分類器 |
| `_service_minutes` 逐分鐘迭代 | 長時間段效能下降 | 改用數學公式計算 |
| I3 使用掛鐘時間（非業務時間） | 假日長時間會誇大 severity | 評估是否改用業務時間 |
| Tripwire 歷史永久有效 | 已解決的退款事件仍永久觸發 | 加入「已解決」標記後可關閉 |
| Agent 生成的 pseudo 資料 | 可能未精確遵循情境設定 | 使用真實標注資料驗證 |

### 假設前提

1. `employees.txt` 中的帳號與 LINE 群組中的發送人名稱**完全一致**（含空格、大小寫）
2. LINE 匯出格式為 **Tab 分隔三欄**（不相容部分 LINE 版本的其他格式）
3. 群組中不存在非員工的內部人員（所有非員工者均視為客戶）

---

## 14. 參數調整指南

### 常見調整場景

**場景 1：業務時間改為 08:00–19:00**
```python
# src/metrics.py
SERVICE_HOURS = (8, 19)
```

**場景 2：降低 I2 的警示敏感度（P90 門檻提高到 3 小時）**
```python
# src/metrics.py :: compute_metrics()
m.i2_severity = _age_to_severity(m.i2_p90_min, warn_min=180, crit_min=600)
```

**場景 3：調整各指標權重（讓情緒問題更重要）**
```python
# main.py 或呼叫 compute_metrics() 時傳入
weights = {"i1": 0.25, "i2": 0.15, "i3": 0.10, "i4": 0.50}
```

**場景 4：新增升級關鍵詞**
```python
# src/enrichment.py
_ESCALATION_KEYWORDS = [
    ...,
    "要媒體曝光",  # 新增
    "consumer protection",  # 英文版本
]
```

**場景 5：更改 LLM 觸發門檻**
```bash
python -X utf8 main.py --llm --llm-threshold 0.15
```

**場景 6：使用更強的模型做 I3 抽取**
```python
# src/llm_extractor.py :: extract_issues()
# 預設 model="claude-haiku-4-5"，改為：
model="claude-opus-4-8"
```

### 指標的 warn / crit 速查表

| 指標 | warn_min | crit_min | 單位 |
|------|----------|----------|------|
| I1 未回應年齡 | 30 | 360 | 業務分鐘 |
| I2 回應延遲 P90 | 120 | 480 | 業務分鐘 |
| I3 未解議題年齡 | 60 | 480 | 掛鐘分鐘 |
| I4 負面比率 warn | 20% | — | — |
| I4 負面比率 crit | — | 60% | — |
| Tripwire I1 硬上限 | — | 480 | 業務分鐘 |

---

*本文件由系統開發者維護，修改任何計算參數或閾值後請同步更新此文件。*
