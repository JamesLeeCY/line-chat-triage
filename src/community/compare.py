"""
Agreement between label sources (human vs Claude vs Qwen …) on shared ids.

Cohen's kappa is reported next to accuracy because "neutral" dominates these
groups: a labeller that says neutral to everything already scores high
accuracy, while its kappa stays near 0.
"""
from typing import Optional

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .gold import read_jsonl


def usable(path: str) -> dict[str, dict]:
    """Latest label per id, minus the ones a human marked unsure."""
    return {k: v for k, v in {r["id"]: r for r in read_jsonl(path)}.items() if not v.get("unsure")}


def agreement(reference: dict[str, dict], candidate: dict[str, dict], field: str,
              ids: Optional[set] = None) -> Optional[dict]:
    shared = sorted(set(reference) & set(candidate) & (ids if ids is not None else set(reference)))
    if not shared:
        return None
    ref = [reference[i][field] for i in shared]
    cand = [candidate[i][field] for i in shared]
    labels = sorted(set(ref) | set(cand))
    return {
        "n": len(shared),
        "accuracy": sum(a == b for a, b in zip(ref, cand)) / len(shared),
        "macro_f1": float(f1_score(ref, cand, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(ref, cand)) if len(set(ref) | set(cand)) > 1 else 1.0,
        "confusion": {"labels": labels, "matrix": confusion_matrix(ref, cand, labels=labels).tolist()},
    }


def format_agreement(name: str, field: str, r: dict) -> str:
    lines = [f"== {name} / {field}  (n={r['n']})  accuracy {r['accuracy']:.3f}  "
             f"macro-F1 {r['macro_f1']:.3f}  kappa {r['kappa']:.3f}",
             "   confusion (row = reference, col = candidate): " + " | ".join(r["confusion"]["labels"])]
    for lbl, row in zip(r["confusion"]["labels"], r["confusion"]["matrix"]):
        lines.append(f"   {lbl:<18}" + "".join(f"{v:>6}" for v in row))
    return "\n".join(lines)
