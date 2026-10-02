"""
Replay a LINE export .txt as signed webhook events against a local webhook,
so the whole ingest path can be exercised without a real LINE group.

LINE would resolve display names via the member API; fake userIds can't be
looked up, so the replay seeds group / member names directly into the DB.

Usage:
    python -X utf8 tools/replay_export.py data/conversations/住宅翻新_陳太太.txt \
        --url http://localhost:8000/callback --db data/line.db
(LINE_CHANNEL_SECRET must match the running webhook.)
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.line_store import LineStore, local_to_ms  # noqa: E402
from src.parser import Message, parse_file  # noqa: E402


def fake_id(prefix: str, name: str) -> str:
    """Deterministic LINE-shaped id (prefix + 32 hex chars)."""
    return prefix + hashlib.md5(name.encode("utf-8")).hexdigest()


def message_to_event(msg: Message, group_id: str) -> dict:
    if msg.is_sticker:
        body = {"type": "sticker", "packageId": "1", "stickerId": "1"}
    elif msg.is_image:
        body = {"type": "image"}
    else:
        body = {"type": "text", "text": msg.text}
    body["id"] = msg.msg_id
    return {
        "type": "message",
        "mode": "active",
        "timestamp": local_to_ms(msg.timestamp),
        "source": {"type": "group", "groupId": group_id, "userId": fake_id("U", msg.sender)},
        "webhookEventId": fake_id("E", msg.msg_id),
        "deliveryContext": {"isRedelivery": False},
        "message": body,
    }


def export_to_events(filepath: str) -> tuple[str, str, list[dict], dict[str, str]]:
    """Return (group_id, group_name, events, {userId: display_name})."""
    group_name, messages = parse_file(filepath, employees=set())
    group_id = fake_id("C", group_name)
    events = [message_to_event(m, group_id) for m in messages]
    members = {fake_id("U", m.sender): m.sender for m in messages}
    return group_id, group_name, events, members


def sign(body: bytes, secret: str) -> str:
    return base64.b64encode(hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()).decode()


def main():
    ap = argparse.ArgumentParser(description="Replay LINE export as webhook events")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="http://localhost:8000/callback")
    ap.add_argument("--db", default=os.environ.get("LINE_DB_PATH", "data/line.db"))
    ap.add_argument("--batch", type=int, default=20, help="events per webhook request")
    args = ap.parse_args()

    secret = os.environ.get("LINE_CHANNEL_SECRET")
    if not secret:
        sys.exit("請設定 LINE_CHANNEL_SECRET（需與 webhook 相同）")

    store = LineStore(args.db)
    for f in args.files:
        group_id, group_name, events, members = export_to_events(f)
        for i in range(0, len(events), args.batch):
            body = json.dumps({"destination": "Ureplay", "events": events[i:i + args.batch]},
                              ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(args.url, data=body, headers={
                "Content-Type": "application/json",
                "X-Line-Signature": sign(body, secret),
            })
            with urllib.request.urlopen(req) as resp:
                resp.read()
        store.upsert_group(group_id, group_name)
        for user_id, name in members.items():
            store.upsert_member(group_id, user_id, name)
        print(f"[replay] {group_name}: {len(events)} events → {args.url}")


if __name__ == "__main__":
    main()
