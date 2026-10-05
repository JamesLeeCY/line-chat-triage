"""
Score a classifier against held-out gold labels.

Reports accuracy and macro-F1 (stance classes are imbalanced, so macro-F1 is
the headline), per-class precision/recall, the confusion matrix, and
calibration (ECE, Brier) — calibration matters because downstream stance
entropy is computed from the predicted probabilities, not just the argmax.
"""
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, bins: int = 10,
                               sample_weight: Optional[np.ndarray] = None) -> float:
    w = np.ones(len(y_idx)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bw = w[mask]
            ece += bw.sum() / w.sum() * abs(np.average(correct[mask], weights=bw) - np.average(conf[mask], weights=bw))
    return float(ece)


def brier_score(proba: np.ndarray, y_idx: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1
    return float(np.average(((proba - onehot) ** 2).sum(axis=1), weights=sample_weight))


def evaluate(classes: Sequence[str], proba: np.ndarray, y_true: Sequence[str],
             sample_weight: Optional[Sequence[float]] = None) -> dict:
    """`sample_weight` (e.g. review-queue stratum weights) turns every metric into a full-split estimate."""
    classes = list(classes)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    known = [y in classes for y in y_true]
    if not all(known):
        raise ValueError(f"test labels missing from training classes: {set(y_true) - set(classes)}")
    y_idx = np.array([classes.index(y) for y in y_true])
    y_pred = [classes[i] for i in proba.argmax(axis=1)]

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0,
                                                       sample_weight=w)
    return {
        "n": len(y_true),
        "weighted": w is not None,
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=w)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0,
                                   sample_weight=w)),
        "ece": expected_calibration_error(proba, y_idx, sample_weight=w),
        "brier": brier_score(proba, y_idx, sample_weight=w),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": float(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion": {"labels": classes,
                      "matrix": np.round(confusion_matrix(y_true, y_pred, labels=classes,
                                                          sample_weight=w), 1).tolist()},
    }


def format_report(name: str, result: dict) -> str:
    lines = [
        f"== {name}  (n={result['n']})" + ("  [weighted: full-split estimate]" if result.get("weighted") else ""),
        f"   accuracy {result['accuracy']:.3f}   macro-F1 {result['macro_f1']:.3f}   "
        f"ECE {result['ece']:.3f}   Brier {result['brier']:.3f}",
        f"   {'class':<18}{'precision':>10}{'recall':>8}{'f1':>8}{'n':>6}",
    ]
    for c, m in result["per_class"].items():
        lines.append(f"   {c:<18}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>8g}")
    labels = result["confusion"]["labels"]
    lines.append("   confusion (row = gold, col = predicted): " + " | ".join(labels))
    for lbl, row in zip(labels, result["confusion"]["matrix"]):
        lines.append(f"   {lbl:<18}" + "".join(f"{v:>8g}" for v in row))
    return "\n".join(lines)
