"""
Agreement between label sources (human vs Claude vs Qwen …) on shared ids.

Cohen's kappa is reported next to accuracy because "neutral" dominates these
groups: a labeller that says neutral to everything already scores high
accuracy, while its kappa stays near 0.
"""
from typing import Optional

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .gold import read_jsonl


def usable(path: str) -> dict[str, dict]:
    """Latest label per id, minus the ones a human marked unsure."""
    return {k: v for k, v in {r["id"]: r for r in read_jsonl(path)}.items() if not v.get("unsure")}


def agreement(reference: dict[str, dict], candidate: dict[str, dict], field: str,
              ids: Optional[set] = None, weights: Optional[dict[str, float]] = None) -> Optional[dict]:
    """
    With `weights` (id -> weight, see review.stratum_weights) only weighted ids
    count, and every metric estimates the full split rather than the queue.
    """
    pool = set(reference) & set(candidate) & (ids if ids is not None else set(reference))
    if weights is not None:
        pool &= set(weights)
    shared = sorted(pool)
    if not shared:
        return None
    ref = [reference[i][field] for i in shared]
    cand = [candidate[i][field] for i in shared]
    w = np.array([weights[i] for i in shared]) if weights is not None else np.ones(len(shared))
    labels = sorted(set(ref) | set(cand))
    return {
        "n": len(shared),
        "weighted": weights is not None,
        "accuracy": float(w[np.array(ref) == np.array(cand)].sum() / w.sum()),
        "macro_f1": float(f1_score(ref, cand, labels=labels, average="macro", zero_division=0, sample_weight=w)),
        "kappa": float(cohen_kappa_score(ref, cand, sample_weight=w)) if len(labels) > 1 else 1.0,
        "confusion": {"labels": labels,
                      "matrix": np.round(confusion_matrix(ref, cand, labels=labels, sample_weight=w), 1).tolist()},
    }


def format_agreement(name: str, field: str, r: dict) -> str:
    tag = "  [weighted: full-split estimate]" if r.get("weighted") else ""
    lines = [f"== {name} / {field}  (n={r['n']}){tag}  accuracy {r['accuracy']:.3f}  "
             f"macro-F1 {r['macro_f1']:.3f}  kappa {r['kappa']:.3f}",
             "   confusion (row = reference, col = candidate): " + " | ".join(r["confusion"]["labels"])]
    for lbl, row in zip(r["confusion"]["labels"], r["confusion"]["matrix"]):
        lines.append(f"   {lbl:<18}" + "".join(f"{v:>8g}" for v in row))
    return "\n".join(lines)
