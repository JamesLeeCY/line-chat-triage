import json
from datetime import datetime
from pathlib import Path

import pytest

from src.enrichment import enrich
from src.line_store import local_to_ms
from src.metrics import compute_metrics
from src.parser import load_employees, merge_fragments, parse_file
from src.telegram_export import find_exports, list_participants, load_export, parse_chat

CONVERSATIONS = sorted(Path("data/conversations").glob("*.txt"))


def _unix(ts: str) -> str:
    """Local (Taiwan) ISO time → export's date_unixtime string."""
    return str(local_to_ms(datetime.fromisoformat(ts)) // 1000)


def _msg(id, ts, sender="客戶A", from_id="user1", text="", **kw):
    return {"id": id, "type": "message", "date": ts, "date_unixtime": _unix(ts),
            "from": sender, "from_id": from_id, "text": text, **kw}


CHAT = {
    "name": "測試群組",
    "type": "private_supergroup",
    "id": 1001,
    "messages": [
        {"id": 1, "type": "service", "date": "2025-09-01T08:59:00", "date_unixtime": _unix("2025-09-01T08:59:00"),
         "actor": "成員1", "actor_id": "user9", "action": "create_group", "text": ""},
        _msg(2, "2025-09-01T09:00:00", text="請問進度如何？"),
        _msg(3, "2025-09-01T09:05:00", "成員1", "user9", text=["請看 ", {"type": "link", "text": "https://x.tw"}],
             reply_to_message_id=2),
        _msg(4, "2025-09-01T09:06:00", file="stickers/s.webp", media_type="sticker", sticker_emoji="👍"),
        _msg(5, "2025-09-01T09:07:00", photo="photos/p.jpg"),
        _msg(6, "2025-09-01T09:08:00", photo="photos/p2.jpg", text="這張圖的尺寸對嗎？"),
        _msg(7, "2025-09-01T09:09:00", file="files/報價.pdf", file_name="報價.pdf"),
        _msg(8, "2025-09-01T09:10:00", file="video_files/v.mp4", media_type="video_file"),
        _msg(9, "2025-09-01T09:11:00", sender=None, from_id="user5", text="帳號已刪除的人"),
        _msg(10, "2025-09-01T09:12:00", text=""),  # empty (e.g. unsupported media) → dropped
    ],
}


# --- parsing ---

def test_parse_chat_content_types_and_roles():
    name, msgs = parse_chat(CHAT, {"成員1"})
    assert name == "測試群組"
    got = [(m.msg_id, m.sender, m.role, m.text, m.is_sticker, m.is_image) for m in msgs]
    assert got == [
        ("tg1001_2", "客戶A", "customer", "請問進度如何？", False, False),
        ("tg1001_3", "成員1", "staff", "請看 https://x.tw", False, False),
        ("tg1001_4", "客戶A", "customer", "", True, False),
        ("tg1001_5", "客戶A", "customer", "", False, True),
        ("tg1001_6", "客戶A", "customer", "這張圖的尺寸對嗎？", False, False),
        ("tg1001_7", "客戶A", "customer", "[檔案] 報價.pdf", False, False),
        ("tg1001_8", "客戶A", "customer", "", False, True),
        ("tg1001_9", "user5", "customer", "帳號已刪除的人", False, False),
    ]
    assert msgs[1].reply_to_msg_id == "tg1001_2"
    assert msgs[0].timestamp == datetime(2025, 9, 1, 9, 0)


def test_staff_matched_by_telegram_id():
    _, msgs = parse_chat(CHAT, {"user9"})
    assert msgs[1].role == "staff"


def test_unixtime_wins_over_local_date_string():
    chat = {**CHAT, "messages": [{**_msg(1, "2025-09-01T09:00:00", text="hi"), "date": "2025-09-01T01:00:00"}]}
    _, msgs = parse_chat(chat, set())
    assert msgs[0].timestamp == datetime(2025, 9, 1, 9, 0)


def test_falls_back_to_date_without_unixtime():
    raw = _msg(1, "2025-09-01T09:00:00", text="hi")
    del raw["date_unixtime"]
    _, msgs = parse_chat({**CHAT, "messages": [raw]}, set())
    assert msgs[0].timestamp == datetime(2025, 9, 1, 9, 0)


def test_list_participants():
    assert list_participants(CHAT)[0] == ("user1", "客戶A", 7)


# --- export files ---

def test_load_export_single_chat_and_full_export(tmp_path):
    single = tmp_path / "a" / "result.json"
    single.parent.mkdir()
    single.write_text(json.dumps(CHAT, ensure_ascii=False), encoding="utf-8")

    full = tmp_path / "b" / "result.json"
    full.parent.mkdir()
    full.write_text(json.dumps({"chats": {"list": [
        CHAT,
        {**CHAT, "name": "私訊", "type": "personal_chat", "id": 2},
        {**CHAT, "name": "頻道", "type": "private_channel", "id": 3},
        {**CHAT, "name": "小群組", "type": "private_group", "id": 4},
    ]}}, ensure_ascii=False), encoding="utf-8")

    assert [c["name"] for c in load_export(str(single))] == ["測試群組"]
    assert [c["name"] for c in load_export(str(full))] == ["測試群組", "小群組"]
    assert find_exports(str(tmp_path)) == [single, full]
    assert find_exports(str(single)) == [single]


# --- end to end: same conversation via LINE export and Telegram export ---

def _line_export_to_telegram(path: Path) -> dict:
    group_name, messages = parse_file(str(path), employees=set())
    tg_messages = []
    for i, m in enumerate(messages, 1):
        raw = _msg(i, m.timestamp.isoformat(), m.sender, f"user{abs(hash(m.sender))}", text=m.text)
        if m.is_sticker:
            raw.update(media_type="sticker", file="stickers/s.webp")
        elif m.is_image:
            raw["photo"] = "photos/p.jpg"
        tg_messages.append(raw)
    return {"name": group_name, "type": "private_supergroup", "id": 1, "messages": tg_messages}


_COMPARED = [
    "i1_severity", "i1_open_questions", "i1_oldest_age_min", "i2_severity", "i2_p90_min",
    "i4_severity", "i4_neg_ratio", "i5_msg_count_24h", "tripwire", "composite",
    "i6_mean_entropy", "i6_entropy_slope", "i6_escalation_fracs",
]


@pytest.mark.parametrize("path", CONVERSATIONS, ids=[p.stem for p in CONVERSATIONS])
def test_telegram_path_matches_line_export(path):
    employees = load_employees("data/employees.txt")
    now = datetime(2025, 9, 5, 17, 0)

    name, line_msgs = parse_file(str(path), employees)
    expected = compute_metrics(name, enrich(merge_fragments(line_msgs)), now=now)

    tg_name, tg_msgs = parse_chat(_line_export_to_telegram(path), employees)
    actual = compute_metrics(tg_name, enrich(merge_fragments(tg_msgs)), now=now)

    assert tg_name == name
    assert [(m.sender, m.role, m.timestamp, m.text) for m in tg_msgs] == \
           [(m.sender, m.role, m.timestamp, m.text) for m in line_msgs]
    for field in _COMPARED:
        assert getattr(actual, field) == getattr(expected, field), field
