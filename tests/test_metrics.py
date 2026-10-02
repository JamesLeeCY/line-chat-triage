from datetime import datetime

import pytest

from src.enrichment import enrich
from src.metrics import _age_to_severity, _service_minutes, compute_metrics


NOW = datetime(2025, 9, 5, 17, 0)  # Friday


# --- business time ---

def test_service_minutes_within_day():
    assert _service_minutes(datetime(2025, 9, 1, 9, 0), datetime(2025, 9, 1, 10, 30)) == 90


def test_service_minutes_skips_night_and_weekend():
    # Fri 17:00 → Mon 10:00 = 1h Friday + 1h Monday
    assert _service_minutes(datetime(2025, 9, 5, 17, 0), datetime(2025, 9, 8, 10, 0)) == 120


def test_service_minutes_reversed_is_zero():
    assert _service_minutes(NOW, datetime(2025, 9, 1)) == 0


@pytest.mark.parametrize("minutes,expected", [(0, 0.0), (29, 0.0), (195, 0.5), (360, 1.0), (999, 1.0)])
def test_age_to_severity(minutes, expected):
    assert _age_to_severity(minutes, warn_min=30, crit_min=360) == pytest.approx(expected)


# --- I1 / I2 ---

def test_unanswered_question_counts_business_minutes(make_msg):
    msgs = enrich([
        make_msg("客戶A", "customer", "2025-09-05T14:00", "請問報價好了嗎？"),
    ])
    m = compute_metrics("g", msgs, now=NOW)
    assert m.i1_open_questions == 1
    assert m.i1_oldest_age_min == 180
    assert 0 < m.i1_severity < 1


def test_answered_question_feeds_latency_not_i1(make_msg):
    msgs = enrich([
        make_msg("客戶A", "customer", "2025-09-05T10:00", "請問報價好了嗎？"),
        make_msg("成員1", "staff", "2025-09-05T13:00", "已寄出"),
    ])
    m = compute_metrics("g", msgs, now=NOW)
    assert m.i1_open_questions == 0
    assert m.i2_p90_min == 180


# --- Tripwire ---

def _escalation_case(make_msg, ts, role="customer", sender="客戶A"):
    return enrich([
        make_msg(sender, role, ts, "這樣我要退款"),
        make_msg("成員1", "staff", ts, "了解，馬上處理"),
    ])


def test_tripwire_fires_on_recent_customer_escalation(make_msg):
    m = compute_metrics("g", _escalation_case(make_msg, "2025-09-04T10:00"), now=NOW)
    assert m.tripwire
    assert m.composite >= 0.95
    assert any("升級詞" in r for r in m.tripwire_reasons)


def test_tripwire_ignores_staff_mentions(make_msg):
    msgs = _escalation_case(make_msg, "2025-09-04T10:00", role="staff", sender="成員2")
    m = compute_metrics("g", msgs, now=NOW)
    assert not m.tripwire


def test_tripwire_expires_after_window(make_msg):
    msgs = _escalation_case(make_msg, "2025-08-20T10:00")
    assert not compute_metrics("g", msgs, now=NOW).tripwire
    # Same history is still flagged when the window is widened
    assert compute_metrics("g", msgs, now=NOW, tripwire_window_hours=24 * 30).tripwire


def test_tripwire_ignores_messages_after_now(make_msg):
    """Backtesting with --now must not see future escalations."""
    m = compute_metrics("g", _escalation_case(make_msg, "2025-09-06T10:00"), now=NOW)
    assert not m.tripwire


def test_tripwire_on_long_unanswered_question(make_msg):
    msgs = enrich([make_msg("客戶A", "customer", "2025-09-01T09:00", "請問什麼時候可以給圖？")])
    m = compute_metrics("g", msgs, now=NOW)
    assert m.tripwire
    assert any("8業務小時" in r for r in m.tripwire_reasons)


# --- I4 ---

def test_negative_sentiment_ratio_uses_72h_window(make_msg):
    msgs = enrich([
        make_msg("客戶A", "customer", "2025-08-01T10:00", "非常失望"),  # outside window
        make_msg("客戶A", "customer", "2025-09-05T10:00", "很失望"),
        make_msg("客戶A", "customer", "2025-09-05T11:00", "好的謝謝"),
    ])
    m = compute_metrics("g", msgs, now=NOW)
    assert m.i4_neg_ratio == pytest.approx(0.5)


@pytest.mark.parametrize("text", ["我要找你們主管談", "我要跟你們老闆談", "找你主管來"])
def test_escalation_keywords_cover_plural_you(make_msg, text):
    msgs = enrich([make_msg("客戶A", "customer", "2025-09-05T10:00", text)])
    assert msgs[0].is_escalation_marker
