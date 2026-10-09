"""
Per-group progress log (data/synth/<run>/progress.jsonl), one line per finished
or failed group, forced to disk before the next group starts. Results already
live in truth.jsonl / pred_*.jsonl; this records how long each group took and
what it cost, so a run interrupted mid-way still shows its pace and retry rate.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path


def log_progress(run_dir, step: str, group: str, **fields) -> None:
    row = {"step": step, "group": group, "finished_at": datetime.now().isoformat(timespec="seconds"), **fields}
    path = Path(run_dir) / "progress.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())                 # survive a crash / power cut right after this group


def call_stats(model, started: float) -> dict:
    """Wall time of one model call plus the token counts Ollama reported, if any."""
    st = getattr(model, "last_stats", None) or {}
    return {"seconds": round(time.time() - started, 1),
            "prompt_tokens": st.get("prompt_eval_count"), "output_tokens": st.get("eval_count")}
