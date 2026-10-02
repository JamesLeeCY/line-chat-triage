from datetime import datetime, timedelta

import pytest

from src.llm_extractor import compute_i3

NOW = datetime(2025, 9, 5, 17, 0)  # naive local (Taiwan) time, like LINE timestamps


def _issue(**kw):
    base = {"status": "unresolved", "raised_at": "", "evidence_msg_ids": []}
    base.update(kw)
    return base


def test_prefers_evidence_message_timestamp(make_msg):
    msg = make_msg("客戶A", "customer", "2025-09-05T15:00", "請問進度？")
    # LLM timestamp is wildly off; evidence message wins
    issues = [_issue(raised_at="2025-09-01T00:00:00", evidence_msg_ids=[msg.msg_id])]
    _, age = compute_i3(issues, NOW, [msg])
    assert age == 120


def test_uses_earliest_evidence_message(make_msg):
    a = make_msg("客戶A", "customer", "2025-09-05T13:00", "第一次問")
    b = make_msg("客戶A", "customer", "2025-09-05T16:00", "再追問")
    _, age = compute_i3([_issue(evidence_msg_ids=[b.msg_id, a.msg_id])], NOW, [a, b])
    assert age == 240


@pytest.mark.parametrize("raised_at", [
    "2025-09-05T15:00:00",        # local, no tz (what the prompt asks for)
    "2025-09-05T07:00:00Z",       # LLM converted to UTC anyway
    "2025-09-05T07:00:00+00:00",
    "2025-09-05T15:00:00+08:00",
])
def test_raised_at_fallback_is_timezone_safe(raised_at):
    _, age = compute_i3([_issue(raised_at=raised_at)], NOW)
    assert age == 120


def test_unknown_evidence_falls_back_to_raised_at(make_msg):
    msg = make_msg("客戶A", "customer", "2025-09-05T10:00", "x")
    _, age = compute_i3([_issue(raised_at="2025-09-05T16:00:00", evidence_msg_ids=["nope"])], NOW, [msg])
    assert age == 60


def test_resolved_and_unparseable_issues_are_ignored():
    issues = [
        _issue(status="resolved", raised_at="2025-09-01T09:00:00"),
        _issue(raised_at="not a date"),
    ]
    assert compute_i3(issues, NOW) == (0.0, 0.0)


@pytest.mark.parametrize("hours_ago,expected", [(0.5, 0.0), (4.5, 0.5), (9, 1.0)])
def test_severity_scale(hours_ago, expected):
    raised = (NOW - timedelta(hours=hours_ago)).isoformat()
    sev, _ = compute_i3([_issue(raised_at=raised)], NOW)
    assert sev == pytest.approx(expected)
