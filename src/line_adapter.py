"""
Convert LINE Messaging API webhook events into LineStore rows.

Only group chats are handled; 1:1 and multi-person room events are ignored.
Media mirrors the LINE export: stickers / images carry no text, other
attachments become a short placeholder text.
"""
import json
from typing import Optional

from .line_store import LineStore

_MEDIA_TYPES = {"image", "video", "audio"}


def _group_source(event: dict) -> Optional[str]:
    source = event.get("source") or {}
    return source.get("groupId") if source.get("type") == "group" else None


def message_text(message: dict) -> tuple[str, bool, bool]:
    """Return (text, is_sticker, is_image) for a webhook message object."""
    mtype = message.get("type")
    if mtype == "text":
        return message.get("text", ""), False, False
    if mtype == "sticker":
        return "", True, False
    if mtype in _MEDIA_TYPES:
        return "", False, True
    if mtype == "file":
        return f"[檔案] {message.get('fileName', '')}".strip(), False, False
    if mtype == "location":
        parts = [message.get("title"), message.get("address")]
        return "[位置] " + " ".join(p for p in parts if p), False, False
    return f"[{mtype}]", False, False


def event_to_row(event: dict) -> Optional[dict]:
    """Map a group message event to a messages-table row, or None if not applicable."""
    group_id = _group_source(event)
    if event.get("type") != "message" or group_id is None:
        return None

    message = event.get("message") or {}
    text, is_sticker, is_image = message_text(message)
    return {
        "message_id": message["id"],
        "group_id": group_id,
        "user_id": (event.get("source") or {}).get("userId"),
        "timestamp_ms": event["timestamp"],
        "text": text,
        "is_sticker": int(is_sticker),
        "is_image": int(is_image),
        "quoted_message_id": message.get("quotedMessageId"),
        "raw_json": json.dumps(event, ensure_ascii=False),
    }


def apply_event(store: LineStore, event: dict) -> None:
    """Persist the effect of one webhook event."""
    etype = event.get("type")
    group_id = _group_source(event)
    if group_id is None:
        return

    if etype == "join":
        store.upsert_group(group_id)
    elif etype == "message":
        row = event_to_row(event)
        store.upsert_group(group_id)
        if row["user_id"]:
            store.upsert_member(group_id, row["user_id"])
        store.add_message(row)
    elif etype == "unsend":
        store.delete_message((event.get("unsend") or {}).get("messageId", ""))
