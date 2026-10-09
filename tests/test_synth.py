import json
import re
from datetime import datetime

import pytest

from src.metrics import _service_minutes as business_minutes_between
from src.parser import load_employees, parse_file
from src.synth.generate import Lines, _prompt_day, generate, render_line_export, role_flips, write_text
from src.synth.predict import LLMVerdict, predict_llm, predict_rule
from src.synth.scenarios import ALERTS, build_scenarios
from src.synth.validate import Fidelity, check_fidelity, score

SCENARIOS = build_scenarios(100, seed=0)


def _dt(sc, slot):
    from datetime import date, timedelta
    d = date.fromisoformat(sc.start) + timedelta(days=slot.day)
    return datetime.fromisoformat(f"{d.isoformat()}T{slot.time}")


def test_scenarios_are_deterministic_and_mixed():
    again = build_scenarios(100, seed=0)
    assert [s.as_dict() for s in again] == [s.as_dict() for s in SCENARIOS]
    levels = {s.risk_level for s in SCENARIOS}
    assert levels == {"low", "medium", "high"}
    assert {a for s in SCENARIOS for a in s.alerts} == set(ALERTS)
    assert len({s.group for s in SCENARIOS}) == 100


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.group)
def test_planted_truth_holds_by_construction(sc):
    now = datetime.fromisoformat(sc.now)
    times = [_dt(sc, s) for s in sc.slots]
    assert times == sorted(times) and all(t.weekday() < 5 for t in times)      # weekdays only
    assert 20 <= len(sc.slots) <= 40
    staff_after = lambda t0: [t for s, t in zip(sc.slots, times) if s.role == "staff" and t0 < t <= now]
    if "unanswered_question" in sc.alerts:
        q = next(t for s, t in zip(sc.slots, times) if s.tag == "unanswered_question")
        assert not staff_after(q) and business_minutes_between(q, now) >= 360
    else:
        # every customer question gets a staff reply before now
        last_customer_q = [t for s, t in zip(sc.slots, times) if s.role == "customer" and "問員工" in s.intent]
        assert all(staff_after(t) for t in last_customer_q)
    if "resolved_escalation" in sc.decoys:
        t = next(t for s, t in zip(sc.slots, times) if s.tag == "resolved_escalation")
        assert (now - t).total_seconds() > 72 * 3600
    if "escalation" in sc.alerts:
        t = next(t for s, t in zip(sc.slots, times) if s.tag == "escalation")
        assert (now - t).total_seconds() < 72 * 3600
    if "after_hours" in sc.decoys:
        q, a = [t for s, t in zip(sc.slots, times) if s.tag == "after_hours"]
        assert q.hour >= 22 and a.hour < 9 and (a - q).total_seconds() < 12 * 3600
    if "slow_response" in sc.alerts:
        assert any(s.tag == "slow_response" for s in sc.slots)


class FakeWriter:
    """Writes each intent back as text, with an explicit threat for escalation slots."""
    model = "fake"

    def ask(self, system, user, schema):
        assert schema is Lines
        rows = [line for line in user.splitlines() if re.match(r"^\d+\.【", line)]
        msgs = []
        for row in rows:
            i = int(row.split(".", 1)[0])
            who = "客戶" if "【客戶說】" in row else "員工"
            text = "我要退款，也要找你們主管" if "揚言" in row else "好的，收到 " + row.split("）", 1)[1][:15]   # intent only, not the sender's display name
            msgs.append({"i": i, "who": who, "text": text})
        return Lines(messages=msgs)


def test_generate_renders_parseable_export_and_truth(tmp_path):
    stats = generate(SCENARIOS[:6], FakeWriter(), str(tmp_path))
    assert stats == {"generated": 6, "skipped": 0, "failed": 0}
    assert generate(SCENARIOS[:6], FakeWriter(), str(tmp_path))["skipped"] == 6        # resumable
    truth = [json.loads(line) for line in (tmp_path / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    employees = load_employees(str(tmp_path / "employees.txt"))
    for t, sc in zip(truth, SCENARIOS[:6]):
        name, msgs = parse_file(str(tmp_path / "conversations" / f"{t['group']}.txt"), employees)
        assert len(msgs) == len(sc.slots)
        assert [m.role for m in msgs] == [s.role for s in sc.slots]
        assert [m.timestamp.strftime("%H:%M") for m in msgs] == [s.time for s in sc.slots]


def test_render_keeps_one_row_per_message():
    sc = SCENARIOS[0]
    text = render_line_export(sc, ["第一行\n第二行"] + ["x"] * (len(sc.slots) - 1))
    assert "第一行 第二行" in text and text.count("\t") == 2 * len(sc.slots)


def test_rule_predictor_and_scoring_end_to_end(tmp_path):
    generate(SCENARIOS[:20], FakeWriter(), str(tmp_path))
    truth = [json.loads(line) for line in (tmp_path / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    preds = predict_rule(str(tmp_path / "conversations"), truth, str(tmp_path / "employees.txt"))
    assert len(preds) == 20 and all(p["risk_level"] in ("low", "medium", "high") for p in preds)
    by = {p["group"]: p for p in preds}
    for t in truth:
        # the fake writer always words threats with tripwire keywords, so the rule system must catch them
        if "escalation" in t["alerts"]:
            assert "escalation" in by[t["group"]]["alerts"]
    s = score(truth, preds)
    assert s["n"] == 20 and 0 <= s["level_accuracy"] <= 1
    assert s["per_alert"]["escalation"]["recall"] == 1.0


def test_score_counts_decoy_false_alarms():
    truth = [{"group": "g1", "risk_level": "low", "alerts": [], "decoys": ["resolved_escalation"]},
             {"group": "g2", "risk_level": "high", "alerts": ["escalation"], "decoys": []}]
    preds = [{"group": "g1", "risk_level": "high", "alerts": ["escalation"], "score": 0.9},
             {"group": "g2", "risk_level": "high", "alerts": ["escalation"], "score": 0.95}]
    s = score(truth, preds)
    assert s["decoys"]["resolved_escalation"] == {"n": 1, "false_alarms": 1}
    assert s["per_alert"]["escalation"] == {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0,
                                             "f1": pytest.approx(2 / 3)}
    assert s["high_risk_auc"] == 1.0 and s["level_accuracy"] == 0.5


class FakeJudge:
    model = "fake"

    def __init__(self, schema, payload):
        self.schema, self.payload, self.calls = schema, payload, 0

    def ask(self, system, user, schema):
        assert schema is self.schema
        self.calls += 1
        return schema(**self.payload)


def test_llm_predictor_and_fidelity_are_resumable(tmp_path):
    generate(SCENARIOS[:3], FakeWriter(), str(tmp_path))
    truth = [json.loads(line) for line in (tmp_path / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    judge = FakeJudge(LLMVerdict, {"reason": "r", "alerts": ["escalation", "escalation"], "risk_level": "high"})
    out = tmp_path / "pred_llm.jsonl"
    rows = predict_llm(str(tmp_path / "conversations"), truth, judge, str(out))
    assert len(rows) == 3 and rows[0]["alerts"] == ["escalation"] and rows[0]["score"] == 1.0
    predict_llm(str(tmp_path / "conversations"), truth, judge, str(out))
    assert judge.calls == 3                                            # second run skipped all

    fid = FakeJudge(Fidelity, {"checks": [], "unplanned_escalation": False, "naturalness": 4, "faithful": True})
    rows = check_fidelity(str(tmp_path), truth, fid, str(tmp_path / "fidelity.jsonl"))
    assert len(rows) == 3 and all(r["faithful"] for r in rows)


def test_day_prompts_mark_who_is_speaking_and_carry_history():
    sc = SCENARIOS[0]
    days = sorted({s.day for s in sc.slots})
    written = {i: f"第{s.day}天的話{i}" for i, s in enumerate(sc.slots)}
    prompts = [_prompt_day(sc, d, written) for d in days]
    assert sum(p.count("【客戶說】") for p in prompts) == sum(s.role == "customer" for s in sc.slots)
    assert sum(p.count("【員工說】") for p in prompts) == sum(s.role == "staff" for s in sc.slots)
    assert "前幾天" not in prompts[0] and "第0天的話" in prompts[1] and "第1天的話" not in prompts[1]
    assert all(("我是客戶" in s.intent) == (s.role == "customer") for s in sc.slots)


def test_role_flips_flags_customer_in_staff_voice():
    sc = SCENARIOS[0]
    own = sc.customer[3:]
    texts = ["x"] * len(sc.slots)
    cust = [i for i, s in enumerate(sc.slots) if s.role == "customer"]
    staff = [i for i, s in enumerate(sc.slots) if s.role == "staff"]
    texts[cust[0]] = f"{own}，非常抱歉，我們會立刻改善"          # pilot failure: apology to themselves
    texts[cust[1]] = "我真的很不滿，再拖我就要退訂金"               # a real customer threat: fine
    texts[staff[0]] = f"{own}您好，我們會盡快處理"                  # staff may address the customer
    texts[staff[1]] = "隔間現在進度到哪了？我這邊要安排家具進場"      # pilot 2: customer line in a staff slot
    assert role_flips(sc, texts) == sorted([cust[0], staff[1]])


class FlippingWriter(FakeWriter):
    """The first `bad_rounds` answers for day 1 put a staff apology in a customer's mouth."""

    def __init__(self, bad_rounds):
        self.bad_rounds, self.calls = bad_rounds, 0

    def ask(self, system, user, schema):
        out = super().ask(system, user, schema)
        if "今天是第1天" in user:
            self.calls += 1
        if "今天是第1天" in user and self.calls <= self.bad_rounds:
            first_customer = next(int(line.split(".", 1)[0]) for line in user.splitlines() if "【客戶說】" in line)
            next(m for m in out.messages if m.i == first_customer).text = "我們會向上回報，請再給我們一點時間"
        return out


def test_write_text_retries_role_flips_and_records_qa():
    sc = SCENARIOS[1]
    n_days = len({s.day for s in sc.slots})
    texts, qa = write_text(sc, FlippingWriter(bad_rounds=1))
    # only the bad day is rewritten: one extra call, not a whole-group redo
    assert qa["attempts"] == n_days + 1 and qa["role_flips"] == [] and len(qa["attempt_log"]) == n_days + 1
    texts, qa = write_text(sc, FlippingWriter(bad_rounds=5), attempts=3)
    assert qa["attempts"] == n_days + 2 and len(qa["role_flips"]) == 1   # day 1 kept after 3 tries, flagged


class WhoSwapWriter(FakeWriter):
    """Echoes the wrong speaker for one slot on the first call: the misalignment pilot 2 hit."""

    def __init__(self):
        self.calls = 0

    def ask(self, system, user, schema):
        self.calls += 1
        out = super().ask(system, user, schema)
        if self.calls == 1:
            m = out.messages[1]
            m.who = "員工" if m.who == "客戶" else "客戶"
        return out


def test_write_text_detects_who_mismatch():
    sc = SCENARIOS[2]
    writer = WhoSwapWriter()
    texts, qa = write_text(sc, writer)
    assert qa["attempt_log"][0]["who_mismatch"] == 1 and qa["attempt_log"][1]["day"] == 0
    assert qa["role_flips"] == []


def test_progress_log_one_line_per_group(tmp_path):
    generate(SCENARIOS[:3], FlippingWriter(bad_rounds=1), str(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["group"] for r in rows] == [s.group for s in SCENARIOS[:3]]
    first = rows[0]
    n_days = len({s.day for s in SCENARIOS[0].slots})
    assert first["step"] == "generate" and first["status"] == "ok" and first["attempts"] == n_days + 1
    assert [a["role_flips"] for a in first["attempt_log"]][:2] == [1, 0] and "seconds" in first
    # the attempt log stays out of the ground-truth file
    truth = json.loads((tmp_path / "truth.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "attempt_log" not in truth["qa"] and truth["qa"]["attempts"] == n_days + 1
    truth_rows = [json.loads(line) for line in (tmp_path / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    predict_llm(str(tmp_path / "conversations"), truth_rows,
                FakeJudge(LLMVerdict, {"reason": "r", "alerts": [], "risk_level": "low"}), str(tmp_path / "pred_llm.jsonl"))
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows].count("predict_llm") == 3


class EmptyWriter(FakeWriter):
    def ask(self, system, user, schema):
        return Lines(messages=[])


def test_progress_log_records_failures(tmp_path):
    stats = generate(SCENARIOS[:1], EmptyWriter(), str(tmp_path))
    assert stats["failed"] == 1
    row = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "failed" and len(row["attempt_log"]) == 3 and "沒有內容" in row["error"]
