"""
Parse Telegram Desktop JSON exports (result.json) into Message lists.

Supports both export kinds:
  - single chat export:   {"name", "type", "id", "messages": [...]}
  - full account export:  {"chats": {"list": [<single chat>, ...]}, ...}
Only group chats are kept; personal chats, bots, channels and saved messages are skipped.

Staff are matched by display name ("from") or Telegram id ("from_id", e.g. "user123456").
Media mirrors the LINE export: stickers / GIFs and uncaptioned photos or videos
carry no text; a caption keeps the message as text so a question isn't lost.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .parser import CONVERSATION_TZ, Message

_GROUP_TYPES = {"private_group", "private_supergroup", "public_supergroup"}
_STICKER_MEDIA = {"sticker", "animation"}
_AV_MEDIA = {"video_file", "video_message", "voice_message", "audio_file"}


def _flatten_text(text) -> str:
    """`text` is either a string or a list of strings / entity dicts."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in text)
    return ""


def _timestamp(msg: dict) -> datetime:
    if msg.get("date_unixtime"):
        return datetime.fromtimestamp(int(msg["date_unixtime"]), tz=CONVERSATION_TZ).replace(tzinfo=None)
    # Older exports only have local time of the exporting machine; assume it matches CONVERSATION_TZ
    return datetime.fromisoformat(msg["date"])


def _content(msg: dict) -> tuple[str, bool, bool]:
    """Return (text, is_sticker, is_image)."""
    text = _flatten_text(msg.get("text")).strip()
    media_type = msg.get("media_type")

    if media_type in _STICKER_MEDIA:
        return "", True, False
    if text:
        return text, False, False
    if "photo" in msg or media_type in _AV_MEDIA:
        return "", False, True
    if "file" in msg:
        name = msg.get("file_name") or Path(str(msg["file"])).name
        return f"[檔案] {name}", False, False
    if "location_information" in msg:
        return "[位置]", False, False
    if "poll" in msg:
        return f"[投票] {(msg['poll'] or {}).get('question', '')}".strip(), False, False
    if "contact_information" in msg:
        return "[聯絡人]", False, False
    return "", False, False


def parse_chat(chat: dict, employees: set[str]) -> tuple[str, list[Message]]:
    """Return (group_name, messages) for one exported chat, in the shape parse_file() produces."""
    group_name = chat.get("name") or f"telegram_{chat.get('id')}"
    prefix = f"tg{chat.get('id', '')}"

    messages: list[Message] = []
    for raw in chat.get("messages", []):
        if raw.get("type") != "message":
            continue  # service messages: joins, pins, title changes …
        text, is_sticker, is_image = _content(raw)
        if not (text or is_sticker or is_image):
            continue

        sender = raw.get("from") or raw.get("from_id") or "?"
        is_staff = raw.get("from") in employees or raw.get("from_id") in employees
        reply_to = raw.get("reply_to_message_id")

        messages.append(Message(
            group_id=group_name,
            msg_id=f"{prefix}_{raw['id']}",
            sender=sender,
            role="staff" if is_staff else "customer",
            timestamp=_timestamp(raw),
            text=text,
            reply_to_msg_id=f"{prefix}_{reply_to}" if reply_to is not None else None,
            is_sticker=is_sticker,
            is_image=is_image,
        ))

    messages.sort(key=lambda m: m.timestamp)  # stable: keeps export order within the same second
    return group_name, messages


def load_export(path: str) -> list[dict]:
    """Return the group chats contained in a result.json (single-chat or full export)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    chats = (data.get("chats") or {}).get("list") if "chats" in data else [data]
    return [c for c in chats or [] if c.get("type") in _GROUP_TYPES]


def find_exports(path: str) -> list[Path]:
    """A result.json file, or every result.json under a directory."""
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(p.rglob("result.json"))


def list_participants(chat: dict) -> list[tuple[Optional[str], str, int]]:
    """(from_id, display_name, message_count) — handy for filling employees.txt."""
    counts: dict[tuple, int] = {}
    for raw in chat.get("messages", []):
        if raw.get("type") == "message":
            key = (raw.get("from_id"), raw.get("from") or "?")
            counts[key] = counts.get(key, 0) + 1
    return sorted(((fid, name, n) for (fid, name), n in counts.items()), key=lambda r: -r[2])


if __name__ == "__main__":
    import sys

    for export in find_exports(sys.argv[1] if len(sys.argv) > 1 else "data/telegram"):
        for chat in load_export(str(export)):
            print(f"\n[{chat.get('name')}]  id={chat.get('id')}  ({export})")
            for from_id, name, n in list_participants(chat):
                print(f"  {from_id}  {name}  ({n} 則)")
