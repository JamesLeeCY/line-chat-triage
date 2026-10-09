"""
Predict the group's own next-window dynamics from the entropy time series
(timeseries.py) — and, above all, measure whether entropy adds anything.

Targets (for window t+1, using only what is known at the end of window t):
- burst:  message volume ≥ BURST_RATIO × its seasonal baseline (median of the
          same hour-of-week slot over the previous SEASON_WEEKS weeks)
- shift:  rarefied topic shift (JS divergence at a fixed sample size) above
          the trailing SHIFT_QUANTILE of past shifts; windows with too few
          entity mentions have no rarefied shift and are dropped
- flip:   net bull/bear sentiment changes sign; windows with too few
          directional messages, or a near-zero net, are dropped

Feature sets are nested so each step's contribution is visible:
  base     seasonality (slot, weekend, market hours next window) + activity lags
  +entropy entity entropy, top-entity share, topic shift, entity coverage
  +stance  directional share, net sentiment (provisional classifier)
plus a persistence baseline (the target's own value at t).

Evaluation is walk-forward (expanding window, TimeSeriesSplit): every
prediction comes from a model fitted on strictly earlier windows. Seasonal
baselines and thresholds are trailing, so nothing leaks from the future.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .timeseries import FREQS, LOCAL_OFFSET

BURST_RATIO = 2.0
SEASON_WEEKS = 4
SHIFT_QUANTILE = 0.8
MIN_DIRECTIONAL = 10       # bull + bear messages per window, for a meaningful sign
MIN_NET = 0.05             # |net sentiment| below this counts as no side

BASE = ["slot_sin", "slot_cos", "weekend", "market_next", "log_vol", "log_vol_lag1", "vol_resid",
        "season_next", "log_speakers", "reply_ratio", "sticker_ratio"]
# rarefied (fixed-sample-size) measures only: the raw ones mostly track mention counts
ENTROPY = ["entity_entropy_rare_f", "topic_shift_rare_f", "topic_missing", "top_entity_share", "entity_coverage"]
STANCE = ["directional_share", "net_sentiment_f"]


def load_series(path: str) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("ts").reset_index(drop=True)


def add_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    width = FREQS[freq]
    df = df.copy()
    local = pd.to_datetime(df["ts"] + LOCAL_OFFSET, unit="s")
    slots_per_day = 86400 // width
    slot_of_day = (local.dt.hour * 3600 + local.dt.minute * 60) // width
    df["slot"] = local.dt.weekday * slots_per_day + slot_of_day          # slot within the week
    angle = 2 * math.pi * slot_of_day / slots_per_day
    df["slot_sin"], df["slot_cos"] = np.sin(angle), np.cos(angle)
    df["weekend"] = (local.dt.weekday >= 5).astype(float)
    end = local + pd.to_timedelta(width, unit="s")
    # window overlaps the TWSE session (Mon–Fri 09:00–13:30); a daily window just needs a weekday
    if freq == "1d":
        in_session = local.dt.weekday < 5
    else:
        in_session = ((local.dt.weekday < 5)
                      & (local.dt.hour * 60 + local.dt.minute < 13 * 60 + 30)
                      & (end.dt.hour * 60 + end.dt.minute > 9 * 60))
    df["market_open"] = in_session.astype(float)
    df["market_next"] = df["market_open"].shift(-1)

    df["log_vol"] = np.log1p(df["n_msgs"])
    df["log_vol_lag1"] = df["log_vol"].shift(1)
    # seasonal baseline: median of the same weekly slot over the previous SEASON_WEEKS weeks (past only)
    df["season"] = (df.groupby("slot")["log_vol"]
                    .transform(lambda s: s.shift(1).rolling(SEASON_WEEKS, min_periods=2).median()))
    df["vol_resid"] = df["log_vol"] - df["season"]
    df["season_next"] = df["season"].shift(-1)                          # known at t: built from past weeks
    df["log_speakers"] = np.log1p(df["n_speakers"])
    df["reply_ratio"] = df["n_replies"] / df["n_msgs"].clip(lower=1)
    df["sticker_ratio"] = df["n_sticker"] / df["n_msgs"].clip(lower=1)

    df["entity_coverage"] = df["n_entity_msgs"] / df["n_text"].clip(lower=1)
    df["topic_missing"] = df["topic_shift_rare"].isna().astype(float)
    for col in ("entity_entropy_rare", "topic_shift_rare"):
        # windows with too few mentions: fill with the past median, flagged by topic_missing
        df[f"{col}_f"] = df[col].fillna(df[col].expanding().median().shift(1)).fillna(df[col].median())
    if "net_sentiment" in df:
        df["net_sentiment_f"] = df["net_sentiment"].fillna(0.0)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # burst at t+1, and its value at t for the persistence baseline
    burst = (df["vol_resid"] >= math.log(BURST_RATIO)).astype(float).where(df["season"].notna())
    df["burst_now"], df["y_burst"] = burst, burst.shift(-1)

    js = df["topic_shift_rare"]          # NaN unless both windows have ≥ RARE_K mentions
    threshold = js.expanding(min_periods=20).quantile(SHIFT_QUANTILE).shift(1)   # past shifts only
    shift = (js > threshold).astype(float).where(js.notna() & threshold.notna())
    df["shift_now"], df["y_shift"] = shift, shift.shift(-1)

    if "net_sentiment" in df:
        sided = ((df["bull"] + df["bear"]) >= MIN_DIRECTIONAL) & (df["net_sentiment"].abs() >= MIN_NET)
        sign = np.sign(df["net_sentiment"]).where(sided)
        prev = sign.ffill().shift(1)                                    # last sided window before t
        flip = (sign != prev).astype(float).where(sign.notna() & prev.notna())
        df["flip_now"], df["y_flip"] = flip, flip.shift(-1)
    return df


@dataclass
class Score:
    n: int
    base_rate: float
    auc: float
    ap: float
    brier: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _score(y: np.ndarray, p: np.ndarray) -> Score:
    if len(np.unique(y)) < 2:
        return Score(len(y), float(y.mean()), float("nan"), float("nan"), float(brier_score_loss(y, p)))
    return Score(len(y), float(y.mean()), float(roc_auc_score(y, p)), float(average_precision_score(y, p)),
                 float(brier_score_loss(y, p)))


def block_bootstrap_delta_auc(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                              reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """
    95% CI of AUC(p_new) − AUC(p_ref) on the same windows, plus P(Δ ≤ 0).
    Resamples contiguous blocks (~5% of the series each) so autocorrelated
    neighbouring windows stay together; an iid bootstrap would be overconfident.
    """
    n = len(y)
    blocks = np.array_split(np.arange(n), max(2, min(20, n // 5)))
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(reps):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.array(deltas) <= 0))


def walk_forward(df: pd.DataFrame, target: str, feature_sets: dict[str, list[str]], persistence: str,
                 n_splits: int = 5) -> dict:
    """Out-of-fold scores per feature set, pooled over the walk-forward test folds."""
    cols = sorted({c for fs in feature_sets.values() for c in fs} | {persistence})
    data = df.dropna(subset=[target] + cols).reset_index(drop=True)
    y = data[target].to_numpy()
    if len(data) < (n_splits + 1) * 10 or len(np.unique(y)) < 2:
        return {"n": len(data), "skipped": "not enough usable windows"}
    preds = {name: np.full(len(data), np.nan) for name in [*feature_sets, "persistence", "base_rate"]}
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(data):
        y_train = y[train_idx]
        preds["base_rate"][test_idx] = y_train.mean()
        # persistence: P(y=1 | value now), estimated on the training windows
        now = data[persistence].to_numpy()
        for v in (0.0, 1.0):
            mask = now[train_idx] == v
            rate = y_train[mask].mean() if mask.any() else y_train.mean()
            preds["persistence"][test_idx[now[test_idx] == v]] = rate
        if len(np.unique(y_train)) < 2:
            for name in feature_sets:
                preds[name][test_idx] = y_train.mean()
            continue
        for name, features in feature_sets.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            model.fit(data.loc[train_idx, features], y_train)
            preds[name][test_idx] = model.predict_proba(data.loc[test_idx, features])[:, 1]
    tested = ~np.isnan(preds["base_rate"])
    scores = {name: _score(y[tested], p[tested]).as_dict() for name, p in preds.items()}
    reference = next(iter(feature_sets))
    if len(np.unique(y[tested])) > 1:
        for name in list(feature_sets)[1:]:
            lo, hi, p_le0 = block_bootstrap_delta_auc(y[tested], preds[name][tested], preds[reference][tested])
            scores[name].update(delta_auc_ci=[round(lo, 4), round(hi, 4)], p_delta_le0=round(p_le0, 4))
    return {"n": int(tested.sum()), "first_test_window": str(data.loc[np.argmax(tested), "window_start"]),
            "reference": reference, "scores": scores}


def run(df: pd.DataFrame, freq: str, n_splits: int = 5) -> dict:
    df = add_targets(add_features(df, freq))
    has_stance = "net_sentiment" in df
    sets = {"base": BASE, "+entropy": BASE + ENTROPY}
    if has_stance:
        sets["+stance"] = BASE + ENTROPY + STANCE
    results = {
        "burst": walk_forward(df, "y_burst", sets, "burst_now", n_splits),
        "shift": walk_forward(df, "y_shift", sets, "shift_now", n_splits),
    }
    if has_stance:
        results["flip"] = walk_forward(df, "y_flip", sets, "flip_now", n_splits)
    return results


def format_results(results: dict) -> str:
    lines = []
    for target, r in results.items():
        if "skipped" in r:
            lines.append(f"== {target}: 略過（可用時間窗 {r['n']}，{r['skipped']}）")
            continue
        base = r["scores"]["base_rate"]["base_rate"]
        lines.append(f"== {target}  (測試時間窗 {r['n']}，自 {r['first_test_window']} 起；正例比例 {base:.1%})")
        lines.append(f"   {'model':<12} {'AUC':>6} {'AP':>6} {'Brier':>7}   ΔAUC vs {r['reference']} [95% CI]")
        for name, s in r["scores"].items():
            ci = (f"   [{s['delta_auc_ci'][0]:+.3f}, {s['delta_auc_ci'][1]:+.3f}]  P(Δ≤0)={s['p_delta_le0']:.3f}"
                  if "delta_auc_ci" in s else "")
            lines.append(f"   {name:<12} {s['auc']:>6.3f} {s['ap']:>6.3f} {s['brier']:>7.4f}{ci}")
    return "\n".join(lines)
