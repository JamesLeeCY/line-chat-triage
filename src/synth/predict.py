"""
Model B: risk level and alerts per group, from the conversation alone.

Three predictors with the same output shape ({group, risk_level, alerts, score}):
- rule:  the existing triage pipeline (I1–I6 metrics + tripwire) at the
         group's own `now`, mapped to alerts by RuleThresholds
- llm:   a local model reads the conversation and answers directly
- combo: per alert, whichever of the two is reliable for it (see `combine`)

All three derive the risk level from their alerts with the ground truth's own
rule, so a level error always traces back to an alert error.

The LLM is told the same business definitions the ground truth uses (not the
rule system's keyword lists), so the comparison is about understanding the
conversation, not matching an implementation.
"""
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .progress import call_stats, log_progress

LEVELS = ("low", "medium", "high")
HIGH_ALERTS = {"unanswered_question", "escalation"}


def level_from_alerts(alerts) -> str:
    """The ground truth's own rule (scenarios._risk_level), so levels are scored on the same definition."""
    alerts = set(alerts)
    if alerts & HIGH_ALERTS or len(alerts) >= 2:
        return "high"
    return "medium" if alerts else "low"


@dataclass(frozen=True)
class RuleThresholds:
    """Severity (0..1, warn → critical in src.metrics) at which a metric raises its alert."""
    unanswered: float = 0.5
    slow: float = 0.5
    negative: float = 0.5
    # "increasingly frustrated" needs more than one negative message: in real groups with five or
    # six customer messages in 72 h, a single ambiguous one already pushes the ratio past 0.10
    min_negative_msgs: int = 2


def rule_metrics(conv_dir: str, truth: list[dict], employees_path: str) -> list[dict]:
    """Run the existing triage pipeline once per group, at the group's own `now`."""
    from ..enrichment import enrich
    from ..metrics import compute_metrics
    from ..parser import load_employees, merge_fragments, parse_file

    employees = load_employees(employees_path)
    out = []
    for t in truth:
        _, messages = parse_file(str(Path(conv_dir) / f"{t['group']}.txt"), employees)
        messages = enrich(merge_fragments(messages, window_secs=60))
        now = datetime.fromisoformat(t["now"])
        m = compute_metrics(t["group"], messages, now=now)
        # same window and cut-off as I4 in src.metrics
        neg_count = sum(1 for msg in messages if msg.role == "customer" and not msg.is_sticker
                        and now - timedelta(hours=72) <= msg.timestamp and (msg.sentiment or 0) < -0.1)
        out.append({"group": t["group"], "i1": m.i1_severity, "i2": m.i2_severity, "i4": m.i4_severity,
                    "neg_count": neg_count, "composite": round(m.composite, 4),
                    # the tripwire also fires on a question unanswered for 8 business hours; only
                    # escalation keywords count as an escalation alert
                    "escalation_marker": any(r.startswith("升級詞") for r in m.tripwire_reasons)})
    return out


def rule_alerts(metrics: list[dict], th: RuleThresholds = RuleThresholds()) -> list[dict]:
    out = []
    for d in metrics:
        alerts = [name for name, on in (
            ("unanswered_question", d["i1"] >= th.unanswered),
            ("slow_response", d["i2"] >= th.slow),
            ("escalation", d["escalation_marker"]),
            ("negative_sentiment", d["i4"] >= th.negative and d["neg_count"] >= th.min_negative_msgs),
        ) if on]
        out.append({"group": d["group"], "risk_level": level_from_alerts(alerts), "alerts": alerts,
                    "score": d["composite"],
                    "detail": {k: d[k] for k in ("i1", "i2", "i4", "neg_count", "escalation_marker")}})
    return out


def predict_rule(conv_dir: str, truth: list[dict], employees_path: str,
                 th: RuleThresholds = RuleThresholds()) -> list[dict]:
    return rule_alerts(rule_metrics(conv_dir, truth, employees_path), th)


# Which predictor each alert is taken from in the combination: the rule system for anything
# that depends on timing (phi4 hallucinated unanswered questions 31 times in the 100-group run),
# the LLM for sentiment (the rule lexicon caught none), either one for escalation.
COMBO_FROM_RULE = {"unanswered_question", "slow_response"}
COMBO_FROM_LLM = {"negative_sentiment"}
COMBO_FROM_EITHER = {"escalation"}


def combine(rule: list[dict], llm: list[dict]) -> list[dict]:
    by_llm = {r["group"]: r for r in llm}
    out = []
    for r in rule:
        m = by_llm.get(r["group"])
        if m is None:
            continue
        ra, la = set(r["alerts"]), set(m["alerts"])
        alerts = sorted((ra & COMBO_FROM_RULE) | (la & COMBO_FROM_LLM) | ((ra | la) & COMBO_FROM_EITHER))
        level = level_from_alerts(alerts)
        # rank by level first, the rule composite breaks ties
        out.append({"group": r["group"], "risk_level": level, "alerts": alerts,
                    "score": LEVELS.index(level) / 2 + r["score"] / 10})
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
        t0 = time.time()
        try:
            v = model.ask(SYSTEM, user, LLMVerdict)
        except Exception as e:
            log_progress(path.parent, "predict_llm", t["group"], status="failed", seconds=round(time.time() - t0, 1),
                         error=str(e))
            print(f"[predict-llm] {t['group']} 失敗：{e}", file=sys.stderr)
            continue
        row = {"group": t["group"], "risk_level": v.risk_level, "alerts": sorted(set(v.alerts)),
               "score": LEVELS.index(v.risk_level) / 2, "reason": v.reason}
        done[t["group"]] = row
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_progress(path.parent, "predict_llm", t["group"], status="ok", **call_stats(model, t0))
        elapsed = time.time() - started
        print(f"[predict-llm] {k}/{len(todo)} {t['group']} → {v.risk_level} {sorted(set(v.alerts))}，"
              f"已花 {elapsed / 60:.0f} 分，預估剩 {elapsed / k * (len(todo) - k) / 60:.0f} 分", file=sys.stderr)
    return [done[t["group"]] for t in truth if t["group"] in done]

