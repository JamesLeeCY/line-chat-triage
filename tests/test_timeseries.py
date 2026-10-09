import math
from collections import Counter

import numpy as np
import pytest

from src.community.entities import extract_entities
from src.community.store import CommunityMessage
from src.community.timeseries import binary_entropy, build_series, entropy_bits, js_divergence

H = 3600
# 2026-01-05 00:00 Asia/Taipei = 2026-01-04 16:00 UTC
DAY0 = 1767542400


@pytest.mark.parametrize("text,expected", [
    ("台積電跟發哥都噴", ["2330", "2454"]),
    ("台積電 2330 台GG", ["2330"]),                # aliases of one entity count once
    ("2330 收 1000", ["2330"]),                    # prices / years are not tickers
    ("2026 年 AI 題材", []),
    ("MUST buy MU", ["MU"]),                       # ASCII tickers are whole tokens only
    ("23300 不是代號", []),
    ("nvda tsla 大盤", ["NVDA", "TSLA", "TAIEX"]),
    ("記憶體 DRAM", ["memory"]),
])
def test_extract_entities(text, expected):
    assert extract_entities(text) == expected


def test_entropy_helpers():
    assert entropy_bits(Counter()) == 0.0
    assert entropy_bits(Counter(a=4)) == 0.0
    assert entropy_bits(Counter(a=1, b=1, c=1, d=1)) == pytest.approx(2.0)
    assert js_divergence(Counter(a=1), Counter(a=5)) == pytest.approx(0.0)
    assert js_divergence(Counter(a=1), Counter(b=1)) == pytest.approx(1.0)
    assert math.isnan(js_divergence(Counter(), Counter(a=1)))
    assert binary_entropy(0.5) == pytest.approx(1.0) and binary_entropy(0.0) == 0.0


class FakeStore:
    def __init__(self, msgs):
        self.msgs = msgs

    def iter_messages(self, group_id, start=None, end=None):
        yield [m for m in self.msgs if m.group_id == group_id]


def _m(i, ts, text, kind="text", sender="u1", reply=None):
    return CommunityMessage(1, i, ts, sender, None, kind, text, reply)


class FakeModel:
    classes_ = ["bearish", "bullish", "neutral"]

    def predict_proba(self, texts):
        # "漲" → bullish 0.6, "跌" → bearish 0.6, else neutral 0.6
        return np.array([[0.6, 0.2, 0.2] if "跌" in t else [0.2, 0.6, 0.2] if "漲" in t else [0.2, 0.2, 0.6]
                         for t in texts])


def test_build_series_hourly_windows_and_columns():
    msgs = [
        _m(1, DAY0 + 10, "台積電要漲", sender="a"),
        _m(2, DAY0 + 20, "發哥跌", sender="b", reply=1),
        _m(3, DAY0 + 30, "", kind="sticker", sender="a"),
        # hour 1 empty; hour 2: same topic as hour 0's 台積電 only
        _m(4, DAY0 + 2 * H + 5, "台積電", sender="c"),
        _m(5, DAY0 + 2 * H + 6, "今天吃什麼", sender="c"),
    ]
    rows = build_series(FakeStore(msgs), 1, freq="1h", stance_model=FakeModel())
    assert [r["window_start"] for r in rows] == ["2026-01-05 00:00", "2026-01-05 01:00", "2026-01-05 02:00"]
    first, empty, third = rows
    assert (first["n_msgs"], first["n_text"], first["n_sticker"], first["n_replies"], first["n_speakers"]) == (3, 2, 1, 1, 2)
    assert first["entity_entropy"] == pytest.approx(1.0) and first["n_entities"] == 2
    assert math.isnan(first["topic_shift_js"])                # nothing before it
    assert (first["bull"], first["bear"], first["net_sentiment"]) == (1, 1, 0)
    assert first["stance_divergence"] == pytest.approx(1.0)
    assert empty["n_msgs"] == 0 and math.isnan(empty["net_sentiment"]) and math.isnan(empty["topic_shift_js"])
    # hour 2 vs the last window that mentioned entities (hour 0), skipping the empty one
    assert third["top_entity"] == "2330" and third["top_entity_share"] == 1.0
    assert 0 < third["topic_shift_js"] < 1
    assert third["directional_share"] == 0.0 and third["bull"] == 0


def test_build_series_daily_aligned_to_local_midnight():
    # 23:30 and 00:30 local fall on different days even though they are 1 h apart
    msgs = [_m(1, DAY0 - 1800, "a"), _m(2, DAY0 + 1800, "b")]
    rows = build_series(FakeStore(msgs), 1, freq="1d")
    assert [r["window_start"] for r in rows] == ["2026-01-04 00:00", "2026-01-05 00:00"]
    assert "bull" not in rows[0]                                # no stance columns without a model


def test_rarefied_removes_sample_size_and_needs_k_mentions():
    from src.community.timeseries import rarefied
    rng = np.random.default_rng(0)
    few = Counter(a=3, b=3)
    assert all(math.isnan(v) for v in rarefied(few, None, rng, k=10))
    # samples of one 8-way distribution at very different sizes: raw entropy is biased low on the
    # small ones, rarefied entropy much less so
    probs = np.full(8, 1 / 8)
    def sample(n):
        return Counter({i: int(c) for i, c in enumerate(rng.multinomial(n, probs)) if c})
    small = [sample(15) for _ in range(40)]
    big = [sample(1500) for _ in range(40)]
    raw_gap = np.mean([entropy_bits(c) for c in big]) - np.mean([entropy_bits(c) for c in small])
    rare_gap = (np.mean([rarefied(c, None, rng, k=10, reps=20)[0] for c in big])
                - np.mean([rarefied(c, None, rng, k=10, reps=20)[0] for c in small]))
    assert raw_gap > 0.25 and abs(rare_gap) < raw_gap / 3
    big = Counter(a=600, b=600)
    # identical distributions shift less than disjoint ones
    _, same = rarefied(big, Counter(a=500, b=500), rng, k=10, reps=50)
    _, disjoint = rarefied(big, Counter(c=50), rng, k=10, reps=50)
    assert disjoint == pytest.approx(1.0) and same < 0.2
