"""
SQLite store for group messages received via the LINE Messaging API webhook.

Timestamps are stored as LINE's epoch milliseconds (UTC) and converted to naive
local time (CONVERSATION_TZ) when loaded, matching what parse_file() produces.

Usage (list groups / members, to fill employees.txt with staff userIds):
    python -m src.line_store data/line.db
"""
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

from .parser import CONVERSATION_TZ, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    group_id   TEXT PRIMARY KEY,
    group_name TEXT
);
CREATE TABLE IF NOT EXISTS members (
    group_id     TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    display_name TEXT,
    PRIMARY KEY (group_id, user_id)
);
CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,
    group_id          TEXT NOT NULL,
    user_id           TEXT,
    timestamp_ms      INTEGER NOT NULL,
    text              TEXT NOT NULL DEFAULT '',
    is_sticker        INTEGER NOT NULL DEFAULT 0,
    is_image          INTEGER NOT NULL DEFAULT 0,
    quoted_message_id TEXT,
    raw_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_group_ts ON messages (group_id, timestamp_ms);
"""


def ms_to_local(ts_ms: int) -> datetime:
    """LINE epoch milliseconds → naive local datetime."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=CONVERSATION_TZ).replace(tzinfo=None)


def local_to_ms(dt: datetime) -> int:
    """Naive local datetime → LINE epoch milliseconds."""
    return int(dt.replace(tzinfo=CONVERSATION_TZ).timestamp() * 1000)


class LineStore:
    def __init__(self, path: str):
        self.path = path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # One connection per operation: the webhook writes from worker threads
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- writes ---

    def add_message(self, row: dict) -> bool:
        """Insert a message row; returns False if it was already stored (redelivery)."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO messages
                   (message_id, group_id, user_id, timestamp_ms, text,
                    is_sticker, is_image, quoted_message_id, raw_json)
                   VALUES (:message_id, :group_id, :user_id, :timestamp_ms, :text,
                           :is_sticker, :is_image, :quoted_message_id, :raw_json)""",
                row,
            )
            return cur.rowcount == 1

    def delete_message(self, message_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))

    def upsert_group(self, group_id: str, group_name: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO groups (group_id, group_name) VALUES (?, ?)
                   ON CONFLICT (group_id) DO UPDATE
                   SET group_name = COALESCE(excluded.group_name, groups.group_name)""",
                (group_id, group_name),
            )

    def upsert_member(self, group_id: str, user_id: str, display_name: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO members (group_id, user_id, display_name) VALUES (?, ?, ?)
                   ON CONFLICT (group_id, user_id) DO UPDATE
                   SET display_name = COALESCE(excluded.display_name, members.display_name)""",
                (group_id, user_id, display_name),
            )

    # --- reads ---

    def groups_missing_name(self) -> list[str]:
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT group_id FROM groups WHERE group_name IS NULL")]

    def members_missing_name(self) -> list[tuple[str, str]]:
        with self._conn() as conn:
            return list(conn.execute("SELECT group_id, user_id FROM members WHERE display_name IS NULL"))

    def list_groups(self) -> list[tuple[str, Optional[str]]]:
        with self._conn() as conn:
            return list(conn.execute("SELECT group_id, group_name FROM groups ORDER BY group_name"))

    def list_members(self) -> list[tuple[str, str, Optional[str], int]]:
        """(group_id, user_id, display_name, message_count)"""
        with self._conn() as conn:
            return list(conn.execute(
                """SELECT m.group_id, m.user_id, m.display_name, COUNT(msg.message_id)
                   FROM members m
                   LEFT JOIN messages msg ON msg.group_id = m.group_id AND msg.user_id = m.user_id
                   GROUP BY m.group_id, m.user_id
                   ORDER BY m.group_id, COUNT(msg.message_id) DESC"""
            ))

    def load_group(self, group_id: str, employees: set[str]) -> tuple[str, list[Message]]:
        """
        Return (group_label, messages) in the same shape as parse_file().
        A sender is staff if their userId or display name is in `employees`.
        """
        with self._conn() as conn:
            name_row = conn.execute("SELECT group_name FROM groups WHERE group_id = ?", (group_id,)).fetchone()
            label = (name_row and name_row[0]) or group_id
            rows = conn.execute(
                """SELECT msg.message_id, msg.user_id, mem.display_name, msg.timestamp_ms, msg.text,
                          msg.is_sticker, msg.is_image, msg.quoted_message_id
                   FROM messages msg
                   LEFT JOIN members mem ON mem.group_id = msg.group_id AND mem.user_id = msg.user_id
                   WHERE msg.group_id = ?
                   ORDER BY msg.timestamp_ms, msg.rowid""",
                (group_id,),
            ).fetchall()

        messages = []
        for message_id, user_id, display_name, ts_ms, text, is_sticker, is_image, quoted in rows:
            is_staff = (user_id in employees) or (display_name in employees)
            messages.append(Message(
                group_id=label,
                msg_id=message_id,
                sender=display_name or user_id or "?",
                role="staff" if is_staff else "customer",
                timestamp=ms_to_local(ts_ms),
                text=text,
                reply_to_msg_id=quoted,
                is_sticker=bool(is_sticker),
                is_image=bool(is_image),
            ))
        return label, messages


def _print_overview(path: str) -> None:
    store = LineStore(path)
    names = dict(store.list_groups())
    current = None
    for group_id, user_id, display_name, count in store.list_members():
        if group_id != current:
            current = group_id
            print(f"\n[{names.get(group_id) or '(未取得群組名稱)'}]  {group_id}")
        print(f"  {user_id}  {display_name or '(未取得名稱)'}  ({count} 則)")


if __name__ == "__main__":
    _print_overview(sys.argv[1] if len(sys.argv) > 1 else "data/line.db")
