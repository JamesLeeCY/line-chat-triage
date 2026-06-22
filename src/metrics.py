"""
Compute I1–I6 metrics for a group's message list.
All severity values are in [0, 1].
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from .parser import Message


SERVICE_HOURS = (9, 18)  # 09:00–18:00, Mon–Fri (0=Mon)
SERVICE_DAYS = {0, 1, 2, 3, 4}  # Mon–Fri


def _in_service(dt: datetime) -> bool:
    return dt.weekday() in SERVICE_DAYS and SERVICE_HOURS[0] <= dt.hour < SERVICE_HOURS[1]


def _service_minutes(start: datetime, end: datetime) -> float:
    """Count business minutes between two datetimes."""
    if end <= start:
        return 0.0
    total = 0.0
    cur = start
    while cur < end:
        if _in_service(cur):
            total += 1
        cur += timedelta(minutes=1)
    return total


def _age_to_severity(minutes: float, warn_min: float = 30, crit_min: float = 360) -> float:
    if minutes < warn_min:
        return 0.0
    if minutes >= crit_min:
        return 1.0
    return (minutes - warn_min) / (crit_min - warn_min)


@dataclass
class GroupMetrics:
    group_id: str
    computed_at: datetime

    # I1 – Unanswered Question Age (max severity among open questions)
    i1_severity: float = 0.0
    i1_open_questions: int = 0
    i1_oldest_age_min: float = 0.0

    # I2 – Customer Response Latency P90
    i2_severity: float = 0.0
    i2_p90_min: float = 0.0

    # I4 – Customer Negative Sentiment
    i4_severity: float = 0.0
    i4_neg_ratio: float = 0.0

    # I3 – Oldest Open Issue Age (from LLM extraction)
    i3_severity: float = 0.0
    i3_oldest_age_min: float = 0.0
    i3_open_issues: int = 0
    i3_issues: list = None  # raw issue dicts from LLM

    # I5 – Volume (message count last 24h)
    i5_msg_count_24h: int = 0

    # Tripwire
    tripwire: bool = False
    tripwire_reasons: list[str] = None

    # Composite score
    composite: float = 0.0

    # I6 – Semantic Entropy (exploratory / diagnostic)
    i6_entropy_series: list = None   # [(timestamp, entropy), ...]
    i6_mean_entropy: float = 0.0
    i6_entropy_slope: float = 0.0    # late_half_mean - early_half_mean; + = entropy rising
    i6_peak_entropy_frac: float = 0.5  # where in conversation the peak occurred (0=start,1=end)
    i6_escalation_fracs: list = None  # fractional positions of escalation msgs in series

    def __post_init__(self):
        if self.tripwire_reasons is None:
            self.tripwire_reasons = []
        if self.i3_issues is None:
            self.i3_issues = []
        if self.i6_entropy_series is None:
            self.i6_entropy_series = []
        if self.i6_escalation_fracs is None:
            self.i6_escalation_fracs = []


def _bigram_entropy(texts: list[str]) -> float:
    """Shannon entropy of character bigrams across a list of texts (bits)."""
    counter: Counter = Counter()
    for text in texts:
        for i in range(len(text) - 1):
            counter[text[i:i+2]] += 1
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def compute_i6_entropy_series(
    messages: list[Message], window: int = 5
) -> tuple[list, float, float, float]:
    """
    Sliding-window semantic entropy across conversation.
    Returns: (series, mean_entropy, slope, peak_frac)
      series       – [(timestamp, entropy_bits), ...]
      slope        – late_half_mean minus early_half_mean; positive = entropy rising toward end
      peak_frac    – 0.0 (start) … 1.0 (end) where maximum entropy occurred
    """
    text_msgs = [m for m in messages if m.text and not m.is_sticker]
    if len(text_msgs) < 2:
        return [], 0.0, 0.0, 0.5

    series = []
    for i, msg in enumerate(text_msgs):
        window_msgs = text_msgs[max(0, i - window + 1): i + 1]
        entropy = _bigram_entropy([m.text for m in window_msgs if m.text])
        series.append((msg.timestamp, entropy))

    entropies = [e for _, e in series]
    mean_e = sum(entropies) / len(entropies)
    mid = len(entropies) // 2
    early_mean = sum(entropies[:mid]) / max(1, mid)
    late_mean = sum(entropies[mid:]) / max(1, len(entropies) - mid)
    slope = late_mean - early_mean
    peak_idx = entropies.index(max(entropies))
    peak_frac = peak_idx / max(1, len(entropies) - 1)

    return series, mean_e, slope, peak_frac


def compute_metrics(
    group_id: str,
    messages: list[Message],
    now: Optional[datetime] = None,
    service_hours: tuple[int, int] = SERVICE_HOURS,
    weights: dict[str, float] = None,
) -> GroupMetrics:
    now = now or datetime.now()
    weights = weights or {"i1": 0.35, "i2": 0.20, "i3": 0.15, "i4": 0.30}
    m = GroupMetrics(group_id=group_id, computed_at=now)

    if not messages:
        return m

    # --- I1: Unanswered question age ---
    open_q_severities = []
    for i, msg in enumerate(messages):
        if msg.role != "customer" or msg.dialogue_act not in ("question", "request"):
            continue
        # Find first staff reply after this message
        responded = False
        for later in messages[i + 1:]:
            if later.role == "staff":
                responded = True
                break
        if not responded:
            age_min = _service_minutes(msg.timestamp, now)
            sev = _age_to_severity(age_min, warn_min=30, crit_min=360)
            open_q_severities.append((sev, age_min))

    if open_q_severities:
        m.i1_open_questions = len(open_q_severities)
        m.i1_severity = max(s for s, _ in open_q_severities)
        m.i1_oldest_age_min = max(a for _, a in open_q_severities)

    # --- I2: Customer response latency P90 ---
    # Cap at 3 calendar days to avoid counting multi-day idle gaps as latency
    _MAX_LAT_DAYS = 3
    latencies = []
    for i, msg in enumerate(messages):
        if msg.role != "customer" or msg.dialogue_act not in ("question", "request"):
            continue
        for later in messages[i + 1:]:
            if (later.timestamp - msg.timestamp).days > _MAX_LAT_DAYS:
                break
            if later.role == "staff":
                lat = _service_minutes(msg.timestamp, later.timestamp)
                latencies.append(lat)
                break

    if latencies:
        latencies.sort()
        p90_idx = int(len(latencies) * 0.9)
        m.i2_p90_min = latencies[min(p90_idx, len(latencies) - 1)]
        m.i2_severity = _age_to_severity(m.i2_p90_min, warn_min=120, crit_min=480)

    # --- I4: Customer negative sentiment ---
    # Use 72h window so recent escalation history isn't lost when crisis just passed
    recent_cutoff = now - timedelta(hours=72)
    customer_msgs = [
        msg for msg in messages
        if msg.role == "customer" and msg.timestamp >= recent_cutoff and not msg.is_sticker
    ]
    if customer_msgs:
        neg_count = sum(1 for msg in customer_msgs if (msg.sentiment or 0) < -0.1)
        m.i4_neg_ratio = neg_count / len(customer_msgs)
        # Severity: >10% negative → starts climbing, >60% → 1.0
        m.i4_severity = min(1.0, max(0.0, (m.i4_neg_ratio - 0.1) / 0.5))

    # --- I5: Volume ---
    cutoff_24h = now - timedelta(hours=24)
    m.i5_msg_count_24h = sum(1 for msg in messages if msg.timestamp >= cutoff_24h)

    # --- Tripwire ---
    reasons = []
    for msg in messages:
        if msg.is_escalation_marker:
            reasons.append(f"升級詞觸發 (msg_id={msg.msg_id}): {msg.text[:30]}")
    if m.i1_oldest_age_min >= 480:  # 8 business hours hard cap
        reasons.append(f"未回應提問超過8業務小時 ({m.i1_oldest_age_min:.0f}分鐘)")
    if reasons:
        m.tripwire = True
        m.tripwire_reasons = reasons

    # --- I6: Semantic Entropy ---
    i6_series, i6_mean, i6_slope, i6_peak_frac = compute_i6_entropy_series(messages)
    m.i6_entropy_series = i6_series
    m.i6_mean_entropy = i6_mean
    m.i6_entropy_slope = i6_slope
    m.i6_peak_entropy_frac = i6_peak_frac
    # Mark where escalation messages fall in the series (fractional position)
    text_msgs = [msg for msg in messages if msg.text and not msg.is_sticker]
    n_text = len(text_msgs)
    if n_text > 1:
        m.i6_escalation_fracs = [
            i / (n_text - 1)
            for i, msg in enumerate(text_msgs)
            if msg.is_escalation_marker
        ]

    # --- Composite ---
    m.composite = (
        weights["i1"] * m.i1_severity
        + weights["i2"] * m.i2_severity
        + weights.get("i3", 0.0) * m.i3_severity
        + weights["i4"] * m.i4_severity
    )
    if m.tripwire:
        m.composite = max(m.composite, 0.95)

    return m
