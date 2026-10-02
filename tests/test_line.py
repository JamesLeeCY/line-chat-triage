import base64
import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.enrichment import enrich
from src.line_adapter import apply_event, event_to_row, message_text
from src.line_store import LineStore, local_to_ms, ms_to_local
from src.line_webhook import create_app, verify_signature
from src.metrics import compute_metrics
from src.parser import load_employees, merge_fragments, parse_file
from tools.replay_export import export_to_events

SECRET = "test-channel-secret"
GROUP = "C" + "0" * 32
CONVERSATIONS = sorted(Path("data/conversations").glob("*.txt"))


def _sign(body: bytes, secret: str = SECRET) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


def _event(msg_id="m1", ts="2025-09-05T10:00", user="Ucustomer", message=None, **kw):
    ev = {
        "type": "message",
        "timestamp": local_to_ms(datetime.fromisoformat(ts)),
        "source": {"type": "group", "groupId": GROUP, "userId": user},
        "message": {"id": msg_id, "type": "text", "text": "請問進度？", **(message or {})},
    }
    ev.update(kw)
    return ev


@pytest.fixture
def store(tmp_path):
    return LineStore(str(tmp_path / "line.db"))


# --- time conversion ---

def test_ms_to_local_is_taiwan_time():
    # 2025-09-05 02:00 UTC == 10:00 in Taiwan
    assert ms_to_local(1757037600000) == datetime(2025, 9, 5, 10, 0)


def test_local_ms_round_trip():
    dt = datetime(2025, 9, 5, 10, 30, 15)
    assert ms_to_local(local_to_ms(dt)) == dt


# --- signature ---

def test_verify_signature():
    body = b'{"events":[]}'
    assert verify_signature(body, _sign(body), SECRET)
    assert not verify_signature(body, _sign(body, "other-secret"), SECRET)
    assert not verify_signature(body + b" ", _sign(body), SECRET)
    assert not verify_signature(body, "", SECRET)


# --- adapter ---

@pytest.mark.parametrize("message,expected", [
    ({"type": "text", "text": "你好"}, ("你好", False, False)),
    ({"type": "sticker"}, ("", True, False)),
    ({"type": "image"}, ("", False, True)),
    ({"type": "video"}, ("", False, True)),
    ({"type": "file", "fileName": "報價單.pdf"}, ("[檔案] 報價單.pdf", False, False)),
    ({"type": "location", "title": "工地", "address": "台北市"}, ("[位置] 工地 台北市", False, False)),
])
def test_message_text(message, expected):
    assert message_text(message) == expected


def test_event_to_row_keeps_quote_and_ignores_non_group():
    row = event_to_row(_event(message={"quotedMessageId": "m0"}))
    assert row["group_id"] == GROUP
    assert row["quoted_message_id"] == "m0"
    assert row["user_id"] == "Ucustomer"

    dm = _event(source={"type": "user", "userId": "Ucustomer"})
    assert event_to_row(dm) is None


# --- store ---

def test_redelivery_is_deduplicated_and_unsend_deletes(store):
    apply_event(store, _event("m1"))
    apply_event(store, _event("m1"))  # LINE redelivery
    apply_event(store, _event("m2", ts="2025-09-05T10:05"))
    _, msgs = store.load_group(GROUP, set())
    assert [m.msg_id for m in msgs] == ["m1", "m2"]

    apply_event(store, {"type": "unsend", "source": {"type": "group", "groupId": GROUP, "userId": "Ucustomer"},
                        "timestamp": 0, "unsend": {"messageId": "m1"}})
    _, msgs = store.load_group(GROUP, set())
    assert [m.msg_id for m in msgs] == ["m2"]


def test_staff_matched_by_user_id_or_display_name(store):
    apply_event(store, _event("m1", user="Ustaff1"))
    apply_event(store, _event("m2", user="Ustaff2"))
    apply_event(store, _event("m3", user="Ucustomer"))
    store.upsert_member(GROUP, "Ustaff2", "成員2")
    store.upsert_group(GROUP, "測試群組")

    label, msgs = store.load_group(GROUP, {"Ustaff1", "成員2"})
    assert label == "測試群組"
    assert [(m.sender, m.role) for m in msgs] == [
        ("Ustaff1", "staff"), ("成員2", "staff"), ("Ucustomer", "customer"),
    ]


def test_upsert_never_erases_known_names(store):
    store.upsert_member(GROUP, "U1", "成員1")
    store.upsert_member(GROUP, "U1")  # a later message event without a name
    assert store.members_missing_name() == []


def test_load_employees_accepts_user_ids_with_comments(tmp_path):
    p = tmp_path / "emp.txt"
    p.write_text("成員1\nU0123456789abcdef0123456789abcdef  # 成員2\n# 整行註解\n", encoding="utf-8")
    assert load_employees(str(p)) == {"成員1", "U0123456789abcdef0123456789abcdef"}


# --- webhook ---

def _post(client, payload, signature=None):
    body = json.dumps(payload).encode()
    return client.post("/callback", content=body, headers={"X-Line-Signature": signature or _sign(body)})


def test_webhook_rejects_bad_signature(store):
    client = TestClient(create_app(SECRET, store))
    resp = _post(client, {"events": [_event()]}, signature="bogus")
    assert resp.status_code == 400
    assert store.load_group(GROUP, set())[1] == []


def test_webhook_verify_request_with_no_events(store):
    """The console's "Verify" button sends an empty, signed event list."""
    client = TestClient(create_app(SECRET, store))
    assert _post(client, {"destination": "Ubot", "events": []}).status_code == 200


def test_webhook_stores_events_and_survives_a_bad_one(store):
    client = TestClient(create_app(SECRET, store))
    bad = {"type": "message", "source": {"type": "group", "groupId": GROUP}, "message": {}}
    resp = _post(client, {"events": [bad, _event("m1")]})
    assert resp.status_code == 200
    assert [m.msg_id for m in store.load_group(GROUP, set())[1]] == ["m1"]


def test_webhook_resolves_names_in_background(store):
    class FakeApi:
        def group_name(self, group_id):
            return "測試群組"

        def member_name(self, group_id, user_id):
            return {"Ucustomer": "客戶A"}.get(user_id)

    client = TestClient(create_app(SECRET, store, api=FakeApi()))
    _post(client, {"events": [_event("m1")]})
    label, msgs = store.load_group(GROUP, set())
    assert label == "測試群組"
    assert msgs[0].sender == "客戶A"


def test_create_app_requires_secret(store):
    with pytest.raises(ValueError):
        create_app("", store)


# --- end to end: export file vs. webhook path give identical metrics ---

_COMPARED = [
    "i1_severity", "i1_open_questions", "i1_oldest_age_min", "i2_severity", "i2_p90_min",
    "i4_severity", "i4_neg_ratio", "i5_msg_count_24h", "tripwire", "tripwire_reasons",
    "composite", "i6_mean_entropy", "i6_entropy_slope", "i6_escalation_fracs",
]


@pytest.mark.parametrize("path", CONVERSATIONS, ids=[p.stem for p in CONVERSATIONS])
def test_webhook_path_matches_export_path(path, store):
    employees = load_employees("data/employees.txt")
    now = datetime(2025, 9, 5, 17, 0)

    group_name, file_msgs = parse_file(str(path), employees)
    expected = compute_metrics(group_name, enrich(merge_fragments(file_msgs)), now=now)

    client = TestClient(create_app(SECRET, store))
    group_id, group_name, events, members = export_to_events(str(path))
    for i in range(0, len(events), 20):
        assert _post(client, {"events": events[i:i + 20]}).status_code == 200
    store.upsert_group(group_id, group_name)
    for user_id, name in members.items():
        store.upsert_member(group_id, user_id, name)

    label, db_msgs = store.load_group(group_id, employees)
    actual = compute_metrics(label, enrich(merge_fragments(db_msgs)), now=now)

    assert label == group_name
    assert [(m.sender, m.role, m.timestamp, m.text) for m in db_msgs] == \
           [(m.sender, m.role, m.timestamp, m.text) for m in file_msgs]
    for field in _COMPARED:
        assert getattr(actual, field) == getattr(expected, field), field
