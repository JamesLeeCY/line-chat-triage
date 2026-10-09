import math

import numpy as np
import pandas as pd
import pytest

from src.community import predict as P

DAY0 = 1767542400            # 2026-01-05 00:00 Asia/Taipei (a Monday)
H4 = 4 * 3600


def _series(n=6 * 7 * 12, seed=0):
    """12 weeks of 4h windows with a daily volume cycle, random bursts and topic shifts."""
    rng = np.random.default_rng(seed)
    slot = np.arange(n) % 6
    vol = (200 + 300 * (slot == 2) + rng.poisson(30, n)).astype(float)
    burst = rng.random(n) < 0.08
    vol[burst] *= 3
    shift = np.clip(rng.normal(0.4, 0.1, n), 0, 1)
    ent_rare = np.clip(rng.normal(2.0, 0.3, n), 0, None)
    # make shift at t+1 depend on entropy at t: the signal +entropy should find
    shift[1:] += 0.25 * (ent_rare[:-1] - 2.0)
    return pd.DataFrame({
        "window_start": [str(i) for i in range(n)], "ts": DAY0 + np.arange(n) * H4,
        "n_msgs": vol, "n_text": vol * 0.9, "n_sticker": vol * 0.05, "n_media": 0, "n_replies": vol * 0.5,
        "n_speakers": vol / 10, "n_entity_msgs": 30, "n_entities": 5, "entity_entropy": ent_rare,
        "top_entity_share": 0.3, "topic_shift_js": shift, "entity_entropy_rare": ent_rare, "topic_shift_rare": shift,
        "heat_surge": rng.exponential(2, n), "entity_volume_growth": rng.normal(0, 0.5, n),
        "new_entity_share": 0.0, "n_new_entities": 0,
        "speaker_entropy_rare": rng.normal(4, 0.3, n), "top_speaker_share": 0.1, "top5_speaker_share": 0.3,
        "new_speaker_share": 0.02,
        "bull": 10, "bear": 10, "directional_share": 0.07, "net_sentiment": rng.normal(0, 0.2, n),
        "stance_divergence": 1.0, "stance_entropy": 1.0,
    })


def test_features_use_only_the_past():
    df = P.add_features(_series(), "4h")
    # seasonal baseline at t must not change when the future is altered
    future_changed = _series()
    future_changed.loc[300:, "n_msgs"] *= 100
    df2 = P.add_features(future_changed, "4h")
    pd.testing.assert_series_equal(df["season"].iloc[:300], df2["season"].iloc[:300])
    assert df["season"].iloc[:6 * 7 * 2 - 1].isna().sum() > 0          # needs ≥ 2 past weeks per slot
    # market flag: Monday 08:00–12:00 window overlaps the session; Saturday never does
    local = pd.to_datetime(df["ts"] + P.LOCAL_OFFSET, unit="s")
    assert df.loc[(local.dt.weekday == 0) & (local.dt.hour == 8), "market_open"].eq(1).all()
    assert df.loc[local.dt.weekday == 5, "market_open"].eq(0).all()


def test_targets_are_next_window_and_thresholds_trailing():
    df = P.add_targets(P.add_features(_series(), "4h"))
    valid = df["y_burst"].notna()
    assert (df.loc[valid, "y_burst"].to_numpy() == df["burst_now"].shift(-1)[valid].to_numpy()).all()
    # the threshold at t needs 20 earlier shifts, so the first target is window 20, predicted from 19
    assert df["y_shift"].iloc[:19].isna().all() and df["y_shift"].iloc[19:].notna().any()
    assert set(df["y_flip"].dropna().unique()) <= {0.0, 1.0}


def test_flip_ignores_unsided_windows():
    s = _series(n=6 * 7 * 3)
    s["net_sentiment"] = [0.3, 0.01, -0.3, 0.3] * (len(s) // 4) + [0.3] * (len(s) % 4)
    df = P.add_targets(P.add_features(s, "4h"))
    # the near-zero window is unsided (NaN) and the next sided window is compared with the last sided one
    assert math.isnan(df["flip_now"].iloc[1]) and df["flip_now"].iloc[2] == 1.0 and df["flip_now"].iloc[3] == 1.0


def test_walk_forward_finds_planted_entropy_signal():
    results = P.run(_series(), "4h")
    shift = results["shift"]["scores"]
    assert shift["+entropy"]["auc"] > shift["base"]["auc"] + 0.05
    assert shift["+entropy"]["delta_auc_ci"][0] > 0
    assert "base" in results["burst"]["scores"]     # synthetic bursts are random: nothing to find
    assert "flip" in results and "+stance" in shift
    assert "== shift" in P.format_results(results)


def test_block_bootstrap_delta():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    good, noise = y + rng.normal(0, 0.5, 400), rng.random(400)
    lo, hi, p = P.block_bootstrap_delta_auc(y, good, noise)
    assert lo > 0 and p == 0
    lo, hi, p = P.block_bootstrap_delta_auc(y, noise, noise)
    assert lo == hi == 0
