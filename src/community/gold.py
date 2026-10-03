"""
Gold-standard labels for the community classifiers.

1. `sample_gold`  — stratified random sample (by group, then evenly by week) of
                    text messages, each with a little preceding context; a fixed
                    share is reserved as the held-out test set.
2. `label_gold`   — Claude labels stance / topic / tickers in batches. Resumable:
                    already-labelled ids are skipped, so a crash or Ctrl+C costs
                    nothing but the in-flight batch.

The labels are the yardstick every classifier (TF-IDF, Jev, …) is scored
against. Backends live in labelers.py (Claude API or a local Ollama model);
human labels from the annotation page are the reference that judges them.
"""
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..parser import CONVERSATION_TZ
from .store import CommunityMessage, CommunityStore

STANCES = ("bullish", "bearish", "neutral")
TOPICS = ("individual_stock", "market_index", "macro", "sector_theme", "trading", "news_info", "chit_chat")

_URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")
_MAX_TEXT = 500


# --- sampling ---

def _week(ts: int) -> str:
    year, week, _ = datetime.fromtimestamp(ts, tz=CONVERSATION_TZ).isocalendar()
    return f"{year}-W{week:02d}"


def _eligible(msg: CommunityMessage) -> bool:
    text = msg.text.strip()
    return len(text) >= 2 and not _URL_ONLY.match(text)


def _context(store: CommunityStore, msg: CommunityMessage) -> dict:
    reply = store.get(msg.group_id, msg.reply_to) if msg.reply_to else None
    return {
        "reply_to_text": reply.text[:_MAX_TEXT] if reply and reply.kind == "text" else None,
        "previous": [m.text[:_MAX_TEXT] for m in store.previous_texts(msg, n=2)],
    }


def sample_gold(
    store: CommunityStore, per_group: dict[int, int], test_size: int, seed: int = 42,
) -> list[dict]:
    """
    Draw `per_group[group_id]` messages from each group, spread evenly across
    ISO weeks (a quiet week gets what it has; the remainder goes to busier ones).
    Weeks rather than months, so a group's partial first / last month doesn't
    get a full month's share.
    Exact duplicate texts are drawn once, so 「噴」×5000 doesn't eat the budget.
    """
    rng = random.Random(seed)
    names = {gid: name for gid, name, *_ in store.groups()}
    records = []

    for group_id, n in per_group.items():
        by_week: dict[str, list[CommunityMessage]] = defaultdict(list)
        for chunk in store.iter_messages(group_id, kind="text"):
            for msg in chunk:
                if _eligible(msg):
                    by_week[_week(msg.ts)].append(msg)

        picked, seen = [], set()
        weeks = sorted(by_week)
        for msgs in by_week.values():
            rng.shuffle(msgs)
        # Round-robin over weeks until the quota is met or every week is exhausted
        cursors = {w: 0 for w in weeks}
        while len(picked) < n and any(cursors[w] < len(by_week[w]) for w in weeks):
            for w in weeks:
                if len(picked) >= n:
                    break
                while cursors[w] < len(by_week[w]):
                    msg = by_week[w][cursors[w]]
                    cursors[w] += 1
                    if msg.text.strip() not in seen:
                        seen.add(msg.text.strip())
                        picked.append(msg)
                        break

        for msg in picked:
            records.append({
                "id": f"{msg.group_id}:{msg.msg_id}",
                "group_id": msg.group_id,
                "group": names.get(msg.group_id, str(msg.group_id)),
                "msg_id": msg.msg_id,
                "ts": msg.ts,
                "text": msg.text[:_MAX_TEXT],
                **_context(store, msg),
            })

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["split"] = "test" if i < test_size else "train"
    return records


# --- labelling ---

class MessageLabel(BaseModel):
    id: str = Field(description="The message id exactly as given")
    stance: Literal["bullish", "bearish", "neutral"]
    topic: Literal["individual_stock", "market_index", "macro", "sector_theme", "trading", "news_info", "chit_chat"]
    tickers: list[str] = Field(description="Stocks / ETFs / indices the message refers to; [] if none")


class LabelBatch(BaseModel):
    labels: list[MessageLabel]


SYSTEM_PROMPT = """\
你是台灣股票社群（Telegram 群組）的訊息標註員。每則訊息請標註三件事，只根據「目標訊息」本身判斷；\
前文與被回覆的訊息只用來理解語意（例如「我也是」「+1」要看它附和的是什麼）。

stance（多空立場）
- bullish：看漲、買進或加碼、持有信心、期待上漲（例：噴、歐印、all in、抱緊、上車、多單、明天漲停）
- bearish：看跌、賣出或減碼、放空、擔心下跌、認賠（例：崩、套牢、要跌了、空單、停損、逃命、韭菜被割）
- neutral：沒有方向性看法：提問、分享資訊但未表態、閒聊、表情、玩笑
- 判斷的是說話者對「後續行情方向」的看法，不是他的部位盈虧。反諷依實際意思判斷（「好棒喔又跌停，明天繼續跌吧」→ bearish）。拿不準時標 neutral。

topic（主題，只選最主要的一個）
- individual_stock：特定個股或 ETF
- market_index：大盤、加權指數、台指期、那斯達克、S&P 500 等整體市場
- macro：總經、利率、Fed、匯率、通膨、政策、地緣政治
- sector_theme：產業或題材（AI、半導體、航運、生技…）而非單一個股
- trading：操作與部位：進出場、停損停利、選擇權、當沖、資金控管
- news_info：轉貼新聞、財報、法說會、公告等資訊分享
- chit_chat：與投資無關的閒聊、問候、玩笑

tickers：訊息提到的標的，台股用代號或常用名稱（2330、台積電、0050），美股用代號（NVDA、TSLA），指數用常用名稱（加權指數、台指期、那斯達克）。沒有就給空陣列。

每則輸入訊息都必須回傳一筆標註，id 原樣照抄。"""


def _format_batch(records: list[dict]) -> str:
    parts = []
    for r in records:
        lines = [f"### id: {r['id']}"]
        if r.get("previous"):
            lines.append("前文：" + " ／ ".join(r["previous"]))
        if r.get("reply_to_text"):
            lines.append("被回覆的訊息：" + r["reply_to_text"])
        lines.append("目標訊息：" + r["text"])
        parts.append("\n".join(lines))
    return "請標註以下訊息：\n\n" + "\n\n".join(parts)


def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_gold(
    sample_path: str, labels_path: str, labeler=None, batch_size: int = 25, workers: int = 4,
    limit: Optional[int] = None, ids: Optional[set] = None,
) -> dict:
    """
    Label every sampled record not yet in `labels_path` with `labeler`
    (anything with `.label(records) -> list[dict]`; default: Claude Haiku).
    `ids` restricts the run to those sample ids (e.g. only the test split).
    """
    if labeler is None:
        from .labelers import ClaudeLabeler
        labeler = ClaudeLabeler()

    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(sample_path)
    done = {r["id"] for r in read_jsonl(labels_path)}
    todo = [r for r in samples if r["id"] not in done and (ids is None or r["id"] in ids)]
    if limit is not None:
        todo = todo[:limit]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    lock = threading.Lock()
    stats = {"requested": len(todo), "labelled": 0, "failed_batches": 0, "missing": 0}
    started = time.time()
    with open(labels_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(labeler.label, b): b for b in batches}
        for i, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            try:
                labels = fut.result()
            except Exception as e:  # keep going; unlabelled ids are retried next run
                stats["failed_batches"] += 1
                print(f"[label] 批次失敗（{len(batch)} 則，下次執行會重試）：{e}", file=sys.stderr)
                continue
            with lock:
                for lbl in labels:
                    out.write(json.dumps(lbl, ensure_ascii=False) + "\n")
                out.flush()
                stats["labelled"] += len(labels)
                stats["missing"] += len(batch) - len(labels)
            elapsed = time.time() - started
            eta = elapsed / i * (len(batches) - i)
            print(f"[label] {i}/{len(batches)} 批完成，累計 {stats['labelled']} 則，"
                  f"已花 {elapsed / 60:.0f} 分，預估剩 {eta / 60:.0f} 分", file=sys.stderr)
    return stats


def load_gold(sample_path: str, labels_path: str) -> list[dict]:
    """Join samples with their labels; unlabelled samples are dropped."""
    labels = {r["id"]: r for r in read_jsonl(labels_path)}
    return [{**s, **labels[s["id"]]} for s in read_jsonl(sample_path) if s["id"] in labels]
