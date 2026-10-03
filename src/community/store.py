"""
SQLite message store for community groups.

A Telegram export is parsed once (`prepare`) and every later step reads from
SQLite in time-ordered chunks, so a 1M-message group never has to be held in
memory or re-parsed from a multi-hundred-MB JSON file.

Timestamps are unix seconds (UTC); convert with CONVERSATION_TZ for display.
"""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

from ..parser import CONVERSATION_TZ
from ..telegram_export import _content, find_exports, load_export

_SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    group_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    type     TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    group_id  INTEGER NOT NULL,
    msg_id    INTEGER NOT NULL,
    ts        INTEGER NOT NULL,
    sender_id TEXT,
    sender    TEXT,
    kind      TEXT NOT NULL,      -- text | sticker | media
    text      TEXT NOT NULL DEFAULT '',
    reply_to  INTEGER,
    PRIMARY KEY (group_id, msg_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_group_ts ON messages (group_id, ts, msg_id);
"""


@dataclass
class CommunityMessage:
    group_id: int
    msg_id: int
    ts: int
    sender_id: Optional[str]
    sender: Optional[str]
    kind: str
    text: str
    reply_to: Optional[int]

    @property
    def local_time(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=CONVERSATION_TZ).replace(tzinfo=None)


_COLUMNS = "group_id, msg_id, ts, sender_id, sender, kind, text, reply_to"


def _ts(raw: dict) -> int:
    if raw.get("date_unixtime"):
        return int(raw["date_unixtime"])
    return int(datetime.fromisoformat(raw["date"]).replace(tzinfo=CONVERSATION_TZ).timestamp())


def _row(group_id: int, raw: dict) -> Optional[tuple]:
    if raw.get("type") != "message":
        return None
    text, is_sticker, is_image = _content(raw)
    if not (text or is_sticker or is_image):
        return None
    kind = "sticker" if is_sticker else "media" if is_image else "text"
    return (
        group_id, raw["id"], _ts(raw), raw.get("from_id"), raw.get("from"),
        kind, text, raw.get("reply_to_message_id"),
    )


class CommunityStore:
    def __init__(self, path: str):
        self.path = path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- import ---

    def import_chat(self, chat: dict) -> int:
        """Insert one exported group chat; re-importing the same export is idempotent."""
        group_id = chat["id"]
        rows = [r for r in (_row(group_id, raw) for raw in chat.get("messages", [])) if r]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO groups (group_id, name, type) VALUES (?, ?, ?) "
                "ON CONFLICT (group_id) DO UPDATE SET name = excluded.name, type = excluded.type",
                (group_id, chat.get("name") or str(group_id), chat.get("type")),
            )
            conn.executemany(f"INSERT OR REPLACE INTO messages ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    # --- reads ---

    def groups(self) -> list[tuple[int, str, int, Optional[int], Optional[int]]]:
        """(group_id, name, message_count, first_ts, last_ts)"""
        with self._conn() as conn:
            return list(conn.execute(
                """SELECT g.group_id, g.name, COUNT(m.msg_id), MIN(m.ts), MAX(m.ts)
                   FROM groups g LEFT JOIN messages m ON m.group_id = g.group_id
                   GROUP BY g.group_id ORDER BY COUNT(m.msg_id) DESC"""
            ))

    def iter_messages(
        self, group_id: int, start: Optional[int] = None, end: Optional[int] = None,
        kind: Optional[str] = None, chunk_size: int = 50_000,
    ) -> Iterator[list[CommunityMessage]]:
        """Yield time-ordered chunks of messages in [start, end)."""
        where, params = ["group_id = ?"], [group_id]
        if start is not None:
            where.append("ts >= ?"); params.append(start)
        if end is not None:
            where.append("ts < ?"); params.append(end)
        if kind is not None:
            where.append("kind = ?"); params.append(kind)
        sql = f"SELECT {_COLUMNS} FROM messages WHERE {' AND '.join(where)} ORDER BY ts, msg_id"

        conn = sqlite3.connect(self.path)
        try:
            cur = conn.execute(sql, params)
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    return
                yield [CommunityMessage(*r) for r in rows]
        finally:
            conn.close()

    def get(self, group_id: int, msg_id: int) -> Optional[CommunityMessage]:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM messages WHERE group_id = ? AND msg_id = ?", (group_id, msg_id)
            ).fetchone()
        return CommunityMessage(*row) if row else None

    def previous_texts(self, msg: CommunityMessage, n: int = 2) -> list[CommunityMessage]:
        """The n text messages right before `msg` in the same group, oldest first."""
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT {_COLUMNS} FROM messages
                    WHERE group_id = ? AND kind = 'text' AND (ts, msg_id) < (?, ?)
                    ORDER BY ts DESC, msg_id DESC LIMIT ?""",
                (msg.group_id, msg.ts, msg.msg_id, n),
            ).fetchall()
        return [CommunityMessage(*r) for r in reversed(rows)]


def prepare(export_path: str, db_path: str) -> list[tuple[str, int]]:
    """Import every group chat under `export_path` into the store. Returns [(name, rows)]."""
    store = CommunityStore(db_path)
    imported = []
    for export in find_exports(export_path):
        for chat in load_export(str(export)):
            imported.append((chat.get("name") or str(chat["id"]), store.import_chat(chat)))
    return imported

