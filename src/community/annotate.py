"""
Local human-annotation server for the gold test split.

    python -X utf8 -m src.community annotate      # then open http://127.0.0.1:8770

Serves one page plus a tiny JSON API; labels are appended to
data/community/labels/human.jsonl (latest entry per id wins), so stopping and
resuming is free. Model labels are deliberately never shown here.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..parser import CONVERSATION_TZ
from .gold import STANCES, TOPICS, read_jsonl

_PAGE = Path(__file__).with_name("annotate.html")


class HumanLabel(BaseModel):
    id: str
    stance: Optional[str] = None
    topic: Optional[str] = None
    unsure: bool = False


def latest_labels(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


def create_app(sample_path: str, labels_path: str, split: str = "test", limit: Optional[int] = 300,
               ids: Optional[list[str]] = None) -> FastAPI:
    """`ids` (e.g. the review queue) fixes exactly which items appear and in what order."""
    samples = read_jsonl(sample_path)
    if ids is not None:
        by_id = {r["id"]: r for r in samples}
        items = [by_id[i] for i in ids if i in by_id]
    else:
        items = [r for r in samples if split in (None, "all") or r["split"] == split]
        if limit:
            items = items[:limit]
    allowed = {r["id"] for r in items}
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Community annotation")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return _PAGE.read_text(encoding="utf-8")

    @app.get("/api/items")
    def get_items():
        def view(r):
            when = datetime.fromtimestamp(r["ts"], tz=CONVERSATION_TZ).strftime("%Y-%m-%d %H:%M")
            return {"id": r["id"], "group": r["group"], "time": when, "text": r["text"],
                    "previous": r.get("previous") or [], "reply_to_text": r.get("reply_to_text")}
        labels = {k: v for k, v in latest_labels(labels_path).items() if k in allowed}
        return {"items": [view(r) for r in items], "labels": labels,
                "stances": list(STANCES), "topics": list(TOPICS)}

    @app.post("/api/labels")
    def save(label: HumanLabel):
        if label.id not in allowed:
            raise HTTPException(404, "unknown id")
        if not label.unsure and (label.stance not in STANCES or label.topic not in TOPICS):
            raise HTTPException(422, "stance and topic are required unless unsure")
        row = {**label.model_dump(), "tickers": [], "annotator": "human", "saved_at": int(time.time())}
        if label.unsure:
            row["stance"] = row["topic"] = None
        with open(labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True, "labelled": len(set(latest_labels(labels_path)) & allowed)}

    return app
