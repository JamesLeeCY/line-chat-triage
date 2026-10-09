"""
Per-window time series of a community group: activity, topic entropy, topic
shift and bull/bear divergence — the inputs for predicting the group's own
dynamics (volume bursts, topic shifts, sentiment flips).

Windows are fixed-length buckets in local time (Taiwan, no DST), so a daily
window is a calendar day. Every column is computed from that window alone,
except the topic shift columns, which compare it with the previous active window.
Raw entropy / shift depend on how many mentions a window has; the `*_rare`
columns redo both on equal-size random draws and are the ones to analyse.

Topic = market entities (entities.py), not free text: entropy over "which
stocks / indices / themes is the group talking about". Low entropy = the room
is fixated on one name; high = attention spread out.

Stance is optional: a text classifier's predicted class per message, counted
per window. Aggregation averages out some per-message noise but not the
classifier's bias — treat it as provisional until labels improve.
"""
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .entities import extract_entities
from .store import CommunityStore

LOCAL_OFFSET = 8 * 3600          # Asia/Taipei, no DST
FREQS = {"1h": 3600, "4h": 4 * 3600, "1d": 86400}
_LOCAL_TZ = timezone(timedelta(seconds=LOCAL_OFFSET))


def entropy_bits(counts) -> float:
    """Shannon entropy (bits) of a count vector / Counter; 0 for empty."""
    values = [v for v in (counts.values() if isinstance(counts, dict) else counts) if v > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum(v / total * math.log2(v / total) for v in values)


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen–Shannon divergence (bits, 0..1) between two count distributions."""
    tp, tq = sum(p.values()), sum(q.values())
    if tp == 0 or tq == 0:
        return float("nan")
    keys = set(p) | set(q)
    pp = {k: p.get(k, 0) / tp for k in keys}
    qq = {k: q.get(k, 0) / tq for k in keys}
    m = {k: (pp[k] + qq[k]) / 2 for k in keys}

    def kl(a):
        return sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a[k] > 0)

    return (kl(pp) + kl(qq)) / 2


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1 or math.isnan(p):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


RARE_K = 10        # mentions drawn per window for sample-size-free entropy / shift
RARE_REPS = 20


def _draw(rng: np.random.Generator, counts: Counter, k: int) -> Counter:
    keys = list(counts)
    picked = rng.multivariate_hypergeometric(np.array([counts[x] for x in keys]), k)
    return Counter({x: int(n) for x, n in zip(keys, picked) if n})


def rarefied(counts: Counter, prev: Optional[Counter], rng: np.random.Generator,
             k: int = RARE_K, reps: int = RARE_REPS) -> tuple[float, float]:
    """
    (entropy, JS shift vs `prev`) averaged over `reps` draws of exactly `k`
    mentions per window, without replacement. Plug-in entropy is biased low
    and JS biased high on small samples, so raw values largely track how many
    mentions a window happened to have; equal-size draws remove that. NaN when
    a window has fewer than `k` mentions.
    """
    nan = float("nan")
    if sum(counts.values()) < k:
        return nan, nan
    usable_prev = prev is not None and sum(prev.values()) >= k
    ents, shifts = [], []
    for _ in range(reps):
        a = _draw(rng, counts, k)
        ents.append(entropy_bits(a))
        if usable_prev:
            shifts.append(js_divergence(a, _draw(rng, prev, k)))
    return float(np.mean(ents)), (float(np.mean(shifts)) if shifts else nan)


HEAT_LOOKBACK = 86400            # heat baseline: the previous day (a week for daily windows)
NEW_LOOKBACK = 7 * 86400         # an entity is "new" if not mentioned in the previous week


def heat_and_novelty(counts: Counter, history: list[Counter], heat_windows: int, new_windows: int) -> dict:
    """
    Topic-heat and new-entity features for one window, from the windows before it.

    heat_surge: the largest Poisson surprise (c − μ) / √(μ + 1) over entities, where μ is
      the entity's mean mentions per window over the last `heat_windows`; how hard the
      fastest-rising name is surging. heat_entity is that name.
    entity_volume_growth: log((mentions now + 1) / (mean mentions per window before + 1)).
    new_entity_share: share of this window's mentions going to entities not mentioned in
      the last `new_windows` (a proportion, so its expectation does not grow with sample size).
    NaN until the full lookback is available.
    """
    nan = float("nan")
    if len(history) < new_windows or len(history) < heat_windows:
        return {"heat_surge": nan, "heat_entity": None, "entity_volume_growth": nan,
                "new_entity_share": nan, "n_new_entities": nan}
    recent = history[-heat_windows:]
    mean = Counter()
    for c in recent:
        mean.update(c)
    total_now = sum(counts.values())
    total_before = sum(mean.values()) / heat_windows
    surge, surge_entity = 0.0, None
    for e, c in counts.items():
        mu = mean.get(e, 0) / heat_windows
        z = (c - mu) / math.sqrt(mu + 1)
        if z > surge:
            surge, surge_entity = z, e
    seen = set()
    for c in history[-new_windows:]:
        seen.update(c)
    new = {e: c for e, c in counts.items() if e not in seen}
    return {
        "heat_surge": surge,
        "heat_entity": surge_entity,
        "entity_volume_growth": math.log((total_now + 1) / (total_before + 1)),
        "new_entity_share": sum(new.values()) / total_now if total_now else 0.0,
        "n_new_entities": len(new),
    }


@dataclass
class _Window:
    n_msgs: int = 0
    n_text: int = 0
    n_sticker: int = 0
    n_media: int = 0
    n_replies: int = 0
    speakers: set = field(default_factory=set)
    entities: Counter = field(default_factory=Counter)
    n_entity_msgs: int = 0
    stance: Optional[np.ndarray] = None   # messages per predicted class


def _bucket(ts: int, width: int) -> int:
    """Window start as unix seconds, aligned to local-time boundaries."""
    return (ts + LOCAL_OFFSET) // width * width - LOCAL_OFFSET


def build_series(store: CommunityStore, group_id: int, freq: str = "1h", stance_model=None,
                 start: Optional[int] = None, end: Optional[int] = None) -> list[dict]:
    """
    One row per window from the group's first to last message, empty windows
    included (zero activity is information for burst prediction).
    `stance_model`: anything with `classes_` and `predict_proba(texts)`.
    """
    width = FREQS[freq]
    windows: dict[int, _Window] = {}
    for chunk in store.iter_messages(group_id, start=start, end=end):
        keys, texts = [], []
        for m in chunk:
            key = _bucket(m.ts, width)
            w = windows.setdefault(key, _Window())
            w.n_msgs += 1
            w.speakers.add(m.sender_id)
            if m.reply_to is not None:
                w.n_replies += 1
            if m.kind == "sticker":
                w.n_sticker += 1
            elif m.kind == "media":
                w.n_media += 1
            if m.kind == "text" and m.text.strip():
                w.n_text += 1
                ents = extract_entities(m.text)
                if ents:
                    w.n_entity_msgs += 1
                    w.entities.update(ents)
                keys.append(key)
                texts.append(m.text)
        if stance_model is not None and texts:
            # one batched call per chunk; per-window calls would be thousands of tiny ones
            # count each message's most likely class: summing probabilities smears a weak
            # classifier's mass over all classes (~29% "directional" vs ~9% by human review),
            # which flattens bull/bear divergence to ~1 bit everywhere
            probs = stance_model.predict_proba(texts)
            probs = np.eye(probs.shape[1])[probs.argmax(axis=1)]
            for key, p in zip(keys, probs):
                w = windows[key]
                w.stance = p.copy() if w.stance is None else w.stance + p

    if not windows:
        return []
    classes = list(stance_model.classes_) if stance_model is not None else []
    rows, prev_entities, history = [], None, []
    rng = np.random.default_rng(0)       # fixed seed: the series is reproducible
    heat_windows = max(1, max(HEAT_LOOKBACK, 7 * 86400 if width >= 86400 else 0) // width)
    new_windows = max(1, NEW_LOOKBACK // width)
    for t in range(min(windows), max(windows) + width, width):
        w = windows.get(t, _Window())
        top, top_n = (w.entities.most_common(1)[0] if w.entities else (None, 0))
        rare_entropy, rare_shift = rarefied(w.entities, prev_entities, rng)
        row = {
            "window_start": datetime.fromtimestamp(t, tz=_LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
            "ts": t,
            "n_msgs": w.n_msgs,
            "n_text": w.n_text,
            "n_sticker": w.n_sticker,
            "n_media": w.n_media,
            "n_replies": w.n_replies,
            "n_speakers": len(w.speakers),
            "n_entity_msgs": w.n_entity_msgs,
            "n_entities": len(w.entities),
            "entity_entropy": entropy_bits(w.entities),
            "top_entity": top,
            "top_entity_share": top_n / sum(w.entities.values()) if w.entities else 0.0,
            # vs the previous window that mentioned anything; NaN when either side is empty
            "topic_shift_js": js_divergence(w.entities, prev_entities) if prev_entities else float("nan"),
            # the same two measures at a fixed sample size (RARE_K mentions); prefer these for analysis
            "entity_entropy_rare": rare_entropy,
            "topic_shift_rare": rare_shift,
            **heat_and_novelty(w.entities, history, heat_windows, new_windows),
        }
        if w.entities:
            prev_entities = w.entities
        history.append(w.entities)
        del history[:-new_windows]
        if classes:
            probs = w.stance if w.stance is not None else np.zeros(len(classes))
            mass = dict(zip(classes, probs))
            bull, bear = mass.get("bullish", 0.0), mass.get("bearish", 0.0)
            directional = bull + bear
            row.update({
                "bull": bull,
                "bear": bear,
                "directional_share": directional / w.n_text if w.n_text else 0.0,
                # -1 all bear .. +1 all bull; NaN without directional mass
                "net_sentiment": (bull - bear) / directional if directional > 0 else float("nan"),
                # bull/bear disagreement: 1 bit = evenly split, 0 = one-sided
                "stance_divergence": binary_entropy(bull / directional) if directional > 0 else 0.0,
                "stance_entropy": entropy_bits(list(probs)),
            })
        rows.append(row)
    return rows

