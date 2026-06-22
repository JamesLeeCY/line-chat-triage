"""
Parse LINE exported chat text files into structured message records.

LINE export format:
  [LINE] Group Name
  儲存日期：YYYY/MM/DD HH:MM

  YYYY.MM.DD 星期X
  HH:MM\tSender\tMessage
  HH:MM\tSender\t[Sticker]
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Message:
    group_id: str
    msg_id: str
    sender: str
    role: str  # staff | customer | system
    timestamp: datetime
    text: str
    reply_to_msg_id: Optional[str] = None
    is_sticker: bool = False
    is_image: bool = False
    # Enrichment fields (filled later)
    dialogue_act: Optional[str] = None
    sentiment: Optional[float] = None
    is_escalation_marker: bool = False


_DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\s+星期")
_MSG_RE = re.compile(r"^(\d{2}):(\d{2})\t(.+?)\t(.+)$")
_SYSTEM_SENDERS = {"", "系統訊息"}


def load_employees(path: str) -> set[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


def parse_file(filepath: str, employees: set[str]) -> tuple[str, list[Message]]:
    """Return (group_id, messages)."""
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8").splitlines()

    group_id = path.stem
    # Try to extract group name from first line
    if lines and lines[0].startswith("[LINE]"):
        group_id = lines[0].replace("[LINE]", "").strip()

    messages: list[Message] = []
    current_date: Optional[datetime] = None
    msg_counter = 0

    for line in lines:
        line = line.rstrip()

        # Date header
        dm = _DATE_RE.match(line)
        if dm:
            current_date = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
            continue

        if current_date is None:
            continue

        # Message line
        mm = _MSG_RE.match(line)
        if not mm:
            continue

        hour, minute, sender, text = int(mm.group(1)), int(mm.group(2)), mm.group(3).strip(), mm.group(4).strip()
        ts = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if sender in _SYSTEM_SENDERS:
            continue

        role = "staff" if sender in employees else "customer"
        is_sticker = text in {"貼圖", "[貼圖]", "Sticker"}
        is_image = text in {"圖片", "[圖片]", "Photo", "Image"}

        msg_counter += 1
        msg_id = f"{group_id}_{msg_counter:05d}"

        messages.append(Message(
            group_id=group_id,
            msg_id=msg_id,
            sender=sender,
            role=role,
            timestamp=ts,
            text="" if (is_sticker or is_image) else text,
            is_sticker=is_sticker,
            is_image=is_image,
        ))

    return group_id, messages


def merge_fragments(messages: list[Message], window_secs: int = 60) -> list[Message]:
    """Merge consecutive messages from same sender within window_secs into one logical message."""
    if not messages:
        return []

    merged: list[Message] = []
    buf = messages[0]

    for msg in messages[1:]:
        same_sender = msg.sender == buf.sender
        within_window = (msg.timestamp - buf.timestamp).total_seconds() <= window_secs
        neither_special = not buf.is_sticker and not buf.is_image and not msg.is_sticker and not msg.is_image

        if same_sender and within_window and neither_special:
            combined = (buf.text + " " + msg.text).strip()
            buf = Message(
                group_id=buf.group_id,
                msg_id=buf.msg_id,
                sender=buf.sender,
                role=buf.role,
                timestamp=buf.timestamp,
                text=combined,
                reply_to_msg_id=buf.reply_to_msg_id,
                is_sticker=False,
                is_image=False,
            )
        else:
            merged.append(buf)
            buf = msg

    merged.append(buf)
    return merged
