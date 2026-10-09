"""
Model B: risk level and alerts per group, from the conversation alone.

Two predictors with the same output shape ({group, risk_level, alerts, score}):
- rule: the existing triage pipeline (I1–I6 metrics + tripwire) at the
  group's own `now`, mapped to alerts by the thresholds below
- llm:  a local model reads the conversation and answers directly

The LLM is told the same business definitions the ground truth uses (not the
rule system's keyword lists), so the comparison is about understanding the
conversation, not matching an implementation.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LEVELS = ("low", "medium", "high")

# rule metrics → alerts; severities are 0..1 (warn → critical) in src.metrics
RULE_ALERT_SEVERITY = 0.5
RULE_HIGH, RULE_MEDIUM = 0.6, 0.3          # composite cut-offs for the risk level


def predict_rule(conv_dir: str, truth: list[dict], employees_path: str) -> list[dict]:
    from ..enrichment import enrich
    from ..metrics import compute_metrics
    from ..parser import load_employees, merge_fragments, parse_file

    employees = load_employees(employees_path)
    out = []
    for t in truth:
        _, messages = parse_file(str(Path(conv_dir) / f"{t['group']}.txt"), employees)
        messages = enrich(merge_fragments(messages, window_secs=60))
        m = compute_metrics(t["group"], messages, now=datetime.fromisoformat(t["now"]))
        alerts = [name for name, on in (
            ("unanswered_question", m.i1_severity >= RULE_ALERT_SEVERITY),
            ("slow_response", m.i2_severity >= RULE_ALERT_SEVERITY),
            ("escalation", m.tripwire),
            ("negative_sentiment", m.i4_severity >= RULE_ALERT_SEVERITY),
        ) if on]
        level = "high" if m.tripwire or m.composite >= RULE_HIGH else "medium" if m.composite >= RULE_MEDIUM else "low"
        out.append({"group": t["group"], "risk_level": level, "alerts": alerts, "score": round(m.composite, 4),
                    "detail": {"i1": m.i1_severity, "i2": m.i2_severity, "i4": m.i4_severity,
                               "tripwire": m.tripwire}})
    return out


SYSTEM = """你是協助主管管理客戶 LINE 工作群組的分析員。你會看到一個群組到「現在時間」為止的對話，發言者名稱以「成員」開頭的是公司員工，其他是客戶。
請判斷這個群組「現在」需要主管注意的程度，並列出成立的警示（可以沒有）：

- unanswered_question：客戶有問題或需求，到現在時間為止員工都沒有回覆，且已超過約 6 個工作小時（工作時間：平日 09:00–18:00）。深夜提問、隔天一早就有回覆的不算。
- slow_response：員工平常回覆客戶就很慢，經常隔好幾個小時才回。
- escalation：最近 72 小時內，客戶揚言退款、投訴、解約、找主管等升級行為。超過 72 小時前發生、之後已經解決且客戶表示滿意的不算。
- negative_sentiment：客戶最近明顯越來越不滿或失望，但沒有到 escalation 的程度。

risk_level：high＝有 unanswered_question 或 escalation，或同時有兩個以上警示；medium＝有一個其他警示；low＝沒有警示。
reason 用一句話說明主要依據。"""


class LLMVerdict(BaseModel):
    reason: str = Field(description="一句話說明主要依據")
    alerts: list[Literal["unanswered_question", "slow_response", "escalation", "negative_sentiment"]]
    risk_level: Literal["low", "medium", "high"]


def predict_llm(conv_dir: str, truth: list[dict], model, out_path: str) -> list[dict]:
    """Resumable: groups already in `out_path` are kept."""
    path = Path(out_path)
    done = {r["group"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} \
        if path.exists() else {}
    todo = [t for t in truth if t["group"] not in done]
    started = time.time()
    for k, t in enumerate(todo, 1):
        text = (Path(conv_dir) / f"{t['group']}.txt").read_text(encoding="utf-8")
        now = datetime.fromisoformat(t["now"])
        user = f"現在時間：{now:%Y/%m/%d %H:%M}（星期{'一二三四五六日'[now.weekday()]}）\n\n對話：\n{text}"
        try:
            v = model.ask(SYSTEM, user, LLMVerdict)
        except Exception as e:
            print(f"[predict-llm] {t['group']} 失敗：{e}", file=sys.stderr)
            continue
        row = {"group": t["group"], "risk_level": v.risk_level, "alerts": sorted(set(v.alerts)),
               "score": LEVELS.index(v.risk_level) / 2, "reason": v.reason}
        done[t["group"]] = row
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elapsed = time.time() - started
        print(f"[predict-llm] {k}/{len(todo)} {t['group']} → {v.risk_level} {sorted(set(v.alerts))}，"
              f"已花 {elapsed / 60:.0f} 分，預估剩 {elapsed / k * (len(todo) - k) / 60:.0f} 分", file=sys.stderr)
    return [done[t["group"]] for t in truth if t["group"] in done]

