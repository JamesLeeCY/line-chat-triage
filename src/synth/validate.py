"""
Model C: validate the synthetic set, then score the predictors against it.

1. Fidelity (needs Claude): does each generated conversation actually carry
   its planted intents? Timing and senders hold by construction; wording may
   not — a "threat" that reads as polite, a decoy that sounds alarming. Groups
   that fail are reported and can be excluded from scoring.
2. Scoring (no model): each predictor vs the planted ground truth — per-alert
   precision / recall, risk-level accuracy and confusion, false alarms on
   decoy groups, and ranking AUC for high-risk groups.

C judges the generator, not the predictors: B is scored against planted
truth, never against another model's opinion.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .scenarios import ALERTS, DECOYS

FIDELITY_SYSTEM = """你是合成資料的品質審查員。你會看到一份「劇本」（每則訊息應表達的意思，以及其中刻意埋入的情境標記）與依劇本生成的 LINE 群組對話。
請檢查生成的對話是否忠實呈現劇本，特別是有標記的情境：

- escalation / resolved_escalation 的升級訊息：客戶是否明確揚言退款、投訴、解約或找主管？
- negative_sentiment：客戶的不滿是否明顯，且「沒有」出現退款、投訴、解約、找主管等升級字眼？
- unanswered_question：客戶是否真的提出需要員工回答的問題，並在之後追問？
- slow_response：員工的回覆內容是否合理（時間由劇本決定，不需檢查）。
- after_hours：深夜的提問語氣是否平和、隔天早上的回覆是否回答了問題？
- resolved_escalation：之後客戶是否明確表示接受、滿意？
- 沒有標記的訊息：是否意外出現升級字眼或強烈不滿，讓群組看起來比劇本更危險？

對每個標記給出是否成立；faithful 只有在所有標記都成立、且沒有意外升級時為 true。"""


class TagCheck(BaseModel):
    tag: str
    realized: bool
    note: str = Field(description="一句話說明")


class Fidelity(BaseModel):
    checks: list[TagCheck]
    unplanned_escalation: bool = Field(description="沒有標記的訊息中是否出現升級字眼或強烈不滿")
    naturalness: int = Field(ge=1, le=5, description="對話自然程度 1–5")
    faithful: bool


def _script(truth: dict) -> str:
    rows = [f"{s['day'] + 1}-{s['time']} {s['speaker']}：{s['intent']}" + (f"  【標記：{s['tag']}】" if s["tag"] else "")
            for s in truth["slots"]]
    return "\n".join(rows)


def check_fidelity(run_dir: str, truth: list[dict], judge, out_path: str) -> list[dict]:
    """Resumable like the predictors; `judge` has `.ask(system, user, schema)` (ClaudeLabeler)."""
    path = Path(out_path)
    done = {r["group"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} \
        if path.exists() else {}
    for t in truth:
        if t["group"] in done:
            continue
        conv = (Path(run_dir) / "conversations" / f"{t['group']}.txt").read_text(encoding="utf-8")
        user = f"劇本：\n{_script(t)}\n\n生成的對話：\n{conv}"
        try:
            f = judge.ask(FIDELITY_SYSTEM, user, Fidelity)
        except Exception as e:
            print(f"[validate] {t['group']} 失敗：{e}", file=sys.stderr)
            continue
        row = {"group": t["group"], **f.model_dump()}
        done[t["group"]] = row
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[validate] {t['group']}：{'通過' if f.faithful else '不符劇本'}（自然度 {f.naturalness}）", file=sys.stderr)
    return [done[t["group"]] for t in truth if t["group"] in done]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    # probability a random high-risk group outranks a random other group (ties count half)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def score(truth: list[dict], preds: list[dict]) -> dict:
    by_group = {p["group"]: p for p in preds}
    rows = [(t, by_group[t["group"]]) for t in truth if t["group"] in by_group]
    if not rows:
        return {"n": 0}
    per_alert = {}
    for a in ALERTS:
        tp = sum(a in t["alerts"] and a in p["alerts"] for t, p in rows)
        fp = sum(a not in t["alerts"] and a in p["alerts"] for t, p in rows)
        fn = sum(a in t["alerts"] and a not in p["alerts"] for t, p in rows)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        per_alert[a] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    levels = ("low", "medium", "high")
    confusion = Counter((t["risk_level"], p["risk_level"]) for t, p in rows)
    decoy = {}
    for d in DECOYS:
        # decoy groups where the decoy's look-alike alert fired although the truth has no such alert
        alike = "escalation" if d == "resolved_escalation" else "unanswered_question"
        groups = [(t, p) for t, p in rows if d in t["decoys"] and alike not in t["alerts"]]
        decoy[d] = {"n": len(groups), "false_alarms": sum(alike in p["alerts"] for _, p in groups)}
    y = np.array([t["risk_level"] == "high" for t, _ in rows], dtype=int)
    s = np.array([p["score"] for _, p in rows], dtype=float)
    return {
        "n": len(rows),
        "level_accuracy": sum(t["risk_level"] == p["risk_level"] for t, p in rows) / len(rows),
        "level_confusion": {f"{a}->{b}": confusion.get((a, b), 0) for a in levels for b in levels},
        "exact_alerts": sum(set(t["alerts"]) == set(p["alerts"]) for t, p in rows) / len(rows),
        "per_alert": per_alert,
        "macro_f1": float(np.mean([v["f1"] for v in per_alert.values()])),
        "decoys": decoy,
        "high_risk_auc": _auc(y, s),
    }


def format_scores(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"== {name}: 沒有可評分的群組"
    lines = [f"== {name}（{s['n']} 個群組）",
             f"   風險等級正確率 {s['level_accuracy']:.0%}｜警示完全正確 {s['exact_alerts']:.0%}｜"
             f"警示 macro-F1 {s['macro_f1']:.2f}｜高風險排序 AUC {s['high_risk_auc']:.2f}",
             f"   {'警示':<20} {'TP':>3} {'FP':>3} {'FN':>3} {'precision':>9} {'recall':>7}"]
    for a, v in s["per_alert"].items():
        lines.append(f"   {a:<20} {v['tp']:>3} {v['fp']:>3} {v['fn']:>3} {v['precision']:>9.2f} {v['recall']:>7.2f}")
    lines.append("   誘餌（不該觸發的）：" + "，".join(
        f"{d} 誤報 {v['false_alarms']}/{v['n']}" for d, v in s["decoys"].items()))
    conf = s["level_confusion"]
    lines.append("   等級混淆（真→預測）：" + "  ".join(f"{k}:{v}" for k, v in conf.items() if v))
    return "\n".join(lines)
