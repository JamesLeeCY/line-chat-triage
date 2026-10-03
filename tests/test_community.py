import json
from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from src.community.classifier import MajorityClass, TfidfLogReg, normalize
from src.community.evaluate import brier_score, evaluate, expected_calibration_error
from src.community.gold import LabelBatch, MessageLabel, _week, label_gold, load_gold, read_jsonl, sample_gold
from src.community.labelers import ClaudeLabeler, OllamaLabeler, make_labeler
from src.community.store import CommunityStore
from src.line_store import local_to_ms

START = datetime(2026, 1, 5, 9, 0)  # a Monday


def _raw(i, ts, text="", sender="甲", **kw):
    return {"id": i, "type": "message", "date": ts.isoformat(), "date_unixtime": str(local_to_ms(ts) // 1000),
            "from": sender, "from_id": f"user{abs(hash(sender)) % 1000}", "text": text, **kw}


def _chat(gid=1, name="台股群", weeks=4, per_day=10):
    msgs, i = [], 0
    for d in range(weeks * 7):
        for k in range(per_day):
            i += 1
            msgs.append(_raw(i, START + timedelta(days=d, minutes=k), text=f"第{d}天 訊息{k} 台積電要噴"))
    msgs.append(_raw(i + 1, START + timedelta(days=1, minutes=30), media_type="sticker", file="s.webp"))
    msgs.append(_raw(i + 2, START + timedelta(days=1, minutes=31), text="+1", reply_to_message_id=1))
    msgs.append({"id": i + 3, "type": "service", "date": START.isoformat(), "action": "join"})
    return {"id": gid, "name": name, "type": "private_supergroup", "messages": msgs}


@pytest.fixture
def store(tmp_path):
    s = CommunityStore(str(tmp_path / "c.db"))
    s.import_chat(_chat())
    s.import_chat(_chat(gid=2, name="美股群", weeks=2, per_day=5))
    return s


# --- store ---

def test_import_kinds_and_idempotency(store):
    gid, name, n, first, last = store.groups()[0]
    assert (gid, name, n) == (1, "台股群", 4 * 7 * 10 + 2)  # service message dropped
    store.import_chat(_chat())  # re-import must not duplicate
    assert store.groups()[0][2] == n
    kinds = Counter(m.kind for chunk in store.iter_messages(1) for m in chunk)
    assert kinds == {"text": 281, "sticker": 1}


def test_iter_messages_chunks_and_window(store):
    chunks = list(store.iter_messages(1, chunk_size=100))
    assert [len(c) for c in chunks] == [100, 100, 82]
    flat = [m for c in chunks for m in c]
    assert [m.ts for m in flat] == sorted(m.ts for m in flat)

    day2 = local_to_ms(START + timedelta(days=2)) // 1000
    day3 = local_to_ms(START + timedelta(days=3)) // 1000
    window = [m for c in store.iter_messages(1, start=day2, end=day3) for m in c]
    assert len(window) == 10 and all(m.local_time.day == 7 for m in window)


def test_previous_texts_skips_non_text_and_respects_order(store):
    reply = store.get(1, 282)  # "+1", replying to msg 1
    prev = store.previous_texts(reply, n=2)
    assert [m.kind for m in prev] == ["text", "text"]
    assert all((m.ts, m.msg_id) < (reply.ts, reply.msg_id) for m in prev)


# --- sampling ---

def test_sample_is_balanced_by_week_and_split(store):
    records = sample_gold(store, {1: 40, 2: 20}, test_size=15, seed=1)
    assert Counter(r["group_id"] for r in records) == {1: 40, 2: 20}
    weeks = Counter(_week(r["ts"]) for r in records if r["group_id"] == 1)
    assert len(weeks) == 4 and max(weeks.values()) - min(weeks.values()) <= 1
    assert Counter(r["split"] for r in records) == {"test": 15, "train": 45}
    for gid in (1, 2):  # duplicate texts are drawn once per group
        texts = [r["text"] for r in records if r["group_id"] == gid]
        assert len(set(texts)) == len(texts)


def test_sample_carries_reply_context(store):
    records = sample_gold(store, {1: 10_000}, test_size=0)
    plus_one = next(r for r in records if r["text"] == "+1")
    assert plus_one["reply_to_text"] == "第0天 訊息0 台積電要噴"
    assert len(plus_one["previous"]) == 2


def test_sample_is_reproducible(store):
    a = sample_gold(store, {1: 30}, test_size=5, seed=7)
    b = sample_gold(store, {1: 30}, test_size=5, seed=7)
    assert a == b


# --- labelling (fake client, no API calls) ---

class FakeMessages:
    def __init__(self, drop_every=None, fail_first=False):
        self.calls, self.drop_every, self.fail_first = [], drop_every, fail_first

    def parse(self, **kw):
        self.calls.append(kw)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("boom")
        ids = [line.split("id: ")[1] for line in kw["messages"][0]["content"].splitlines() if line.startswith("### id: ")]
        labels = [MessageLabel(id=i, stance="bullish", topic="individual_stock", tickers=["2330"])
                  for n, i in enumerate(ids) if not (self.drop_every and n % self.drop_every == 0)]
        labels.append(MessageLabel(id="not-asked", stance="neutral", topic="chit_chat", tickers=[]))
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=LabelBatch(labels=labels))


def _write_sample(store, tmp_path, n=30):
    path = tmp_path / "sample.jsonl"
    records = sample_gold(store, {1: n}, test_size=5)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return str(path)


def test_label_gold_is_resumable(store, tmp_path):
    sample, labels = _write_sample(store, tmp_path), str(tmp_path / "labels.jsonl")
    client = SimpleNamespace(messages=FakeMessages())

    stats = label_gold(sample, labels, labeler=ClaudeLabeler(client=client), batch_size=10, workers=2, limit=12)
    assert stats["labelled"] == 12 and len(read_jsonl(labels)) == 12
    first = client.messages.calls[0]
    assert first["model"] == "claude-haiku-4-5"
    assert "output_config" not in first  # Haiku 4.5 rejects effort
    assert first["output_format"] is LabelBatch

    stats = label_gold(sample, labels, labeler=ClaudeLabeler(client=client), batch_size=10, workers=2)
    assert stats["requested"] == 18
    ids = [r["id"] for r in read_jsonl(labels)]
    assert len(ids) == len(set(ids)) == 30
    assert "not-asked" not in ids


def test_effort_only_sent_when_requested(store, tmp_path):
    sample, labels = _write_sample(store, tmp_path, n=10), str(tmp_path / "labels.jsonl")
    client = SimpleNamespace(messages=FakeMessages())
    label_gold(sample, labels, labeler=ClaudeLabeler("claude-opus-5-5", effort="medium", client=client), batch_size=10)
    assert client.messages.calls[0]["output_config"] == {"effort": "medium"}
    assert client.messages.calls[0]["model"] == "claude-opus-5-5"


def test_label_gold_survives_failed_batch_and_missing_ids(store, tmp_path):
    sample, labels = _write_sample(store, tmp_path), str(tmp_path / "labels.jsonl")
    stats = label_gold(sample, labels, labeler=ClaudeLabeler(client=SimpleNamespace(messages=FakeMessages(drop_every=5, fail_first=True))),
                       batch_size=10, workers=1)
    assert stats["failed_batches"] == 1
    assert stats["missing"] == 4  # 2 batches × ids at position 0 and 5
    # The next run only asks for what is still unlabelled
    stats = label_gold(sample, labels, labeler=ClaudeLabeler(client=SimpleNamespace(messages=FakeMessages())), batch_size=10, workers=1)
    assert stats["requested"] == 14
    assert len(load_gold(sample, labels)) == 30


def test_refusal_is_a_failed_batch(store, tmp_path):
    class Refusing:
        def parse(self, **kw):
            return SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="x"), parsed_output=None)

    sample, labels = _write_sample(store, tmp_path, n=10), str(tmp_path / "labels.jsonl")
    stats = label_gold(sample, labels, labeler=ClaudeLabeler(client=SimpleNamespace(messages=Refusing())), batch_size=10)
    assert stats["failed_batches"] == 1 and read_jsonl(labels) == []


# --- classifier & evaluation ---

BULL = ["台積電要噴了", "歐印 2330", "明天漲停", "抱緊上車", "多單續抱", "噴噴噴", "加碼 nvda", "all in 0050"]
BEAR = ["要崩了快逃", "套牢了", "停損出場", "空單進場", "明天跌停", "崩崩崩", "認賠殺出", "韭菜又被割"]
NEUT = ["大家早安", "請問這是什麼", "吃飯了嗎", "哈哈哈", "今天天氣好", "有人在嗎", "晚安", "週末愉快"]


def test_normalize():
    assert normalize("  NVDA  看 https://x.com/a  噴 ") == "nvda 看 url 噴"


def test_tfidf_learns_slang_and_outputs_probabilities():
    X = BULL * 3 + BEAR * 3 + NEUT * 3
    y = ["bullish"] * 24 + ["bearish"] * 24 + ["neutral"] * 24
    model = TfidfLogReg(min_df=1).fit(X, y)
    assert model.classes_ == ["bearish", "bullish", "neutral"]
    proba = model.predict_proba(["噴噴", "崩了", "早安大家"])
    assert np.allclose(proba.sum(axis=1), 1)
    assert model.predict(["噴噴", "崩了", "早安大家"]) == ["bullish", "bearish", "neutral"]
    assert any("噴" in f for f in model.top_features(5)["bullish"])


def test_majority_baseline():
    m = MajorityClass().fit(["a"] * 4, ["neutral", "neutral", "neutral", "bullish"])
    assert m.classes_ == ["bullish", "neutral"]
    assert np.allclose(m.predict_proba(["x", "y"]), [[0.25, 0.75]] * 2)


def test_calibration_metrics():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, y) == 0
    assert brier_score(perfect, y) == 0
    overconfident_wrong = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(overconfident_wrong, y) == pytest.approx(1.0)
    assert brier_score(overconfident_wrong, y) == pytest.approx(2.0)


def test_evaluate_report_shape():
    proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    r = evaluate(["bearish", "bullish"], proba, ["bearish", "bullish", "bullish"])
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["confusion"]["matrix"] == [[1, 0], [1, 1]]
    assert r["per_class"]["bullish"]["recall"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        evaluate(["bearish", "bullish"], proba, ["bearish", "neutral", "bullish"])


# --- ollama backend (urlopen mocked, no server needed) ---

class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_labeler_payload_and_parsing(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        content = LabelBatch(labels=[
            MessageLabel(id="a", stance="bearish", topic="market_index", tickers=["加權指數"]),
            MessageLabel(id="a", stance="bullish", topic="chit_chat", tickers=[]),  # duplicate: first wins
            MessageLabel(id="zzz", stance="neutral", topic="chit_chat", tickers=[]),  # never asked
        ]).model_dump_json()
        return _FakeResponse({"message": {"content": content}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = OllamaLabeler("qwen3:8b").label([{"id": "a", "text": "要崩了"}, {"id": "b", "text": "早"}])

    assert sent["url"] == "http://localhost:11434/api/chat"
    body = sent["body"]
    assert body["model"] == "qwen3:8b" and body["stream"] is False and body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["format"]["properties"]["labels"]["type"] == "array"
    assert "目標訊息：要崩了" in body["messages"][1]["content"]
    assert out == [{"id": "a", "stance": "bearish", "topic": "market_index", "tickers": ["加權指數"]}]


def test_ollama_unreachable_is_a_clear_error(monkeypatch):
    import urllib.error

    def refuse(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(RuntimeError, match="連不到 Ollama"):
        OllamaLabeler().label([{"id": "a", "text": "x"}])


def test_make_labeler_think_flag_by_model():
    assert make_labeler("ollama").think is False                      # qwen3 default: thinking off
    assert make_labeler("ollama", "qwen2.5:7b-instruct").think is None  # no thinking mode: don't send
    assert "think" not in make_labeler("ollama", "qwen2.5:7b-instruct")._payload([{"id": "a", "text": "x"}])
    assert make_labeler("ollama").name == "qwen3-8b"
    with pytest.raises(ValueError):
        make_labeler("openai")


# --- human annotation API ---

def test_annotate_api_saves_and_validates(store, tmp_path):
    from fastapi.testclient import TestClient

    from src.community.annotate import create_app

    sample = _write_sample(store, tmp_path, n=20)
    labels = str(tmp_path / "labels" / "human.jsonl")
    client = TestClient(create_app(sample, labels, split="test", limit=3))

    assert "多空標註" in client.get("/").text
    data = client.get("/api/items").json()
    ids = [it["id"] for it in data["items"]]
    assert len(ids) == 3 and data["labels"] == {} and "previous" in data["items"][0]

    assert client.post("/api/labels", json={"id": ids[0], "stance": "bullish", "topic": "trading"}).json()["labelled"] == 1
    # relabel: latest wins, still one labelled item
    assert client.post("/api/labels", json={"id": ids[0], "stance": "bearish", "topic": "trading"}).json()["labelled"] == 1
    assert client.post("/api/labels", json={"id": ids[1], "unsure": True}).json()["labelled"] == 2
    assert client.post("/api/labels", json={"id": ids[2], "stance": "moon", "topic": "trading"}).status_code == 422
    assert client.post("/api/labels", json={"id": "nope", "stance": "bullish", "topic": "trading"}).status_code == 404

    saved = client.get("/api/items").json()["labels"]
    assert saved[ids[0]]["stance"] == "bearish" and saved[ids[1]]["unsure"] is True


# --- agreement between label sources ---

def test_compare_agreement_skips_unsure(tmp_path):
    from src.community.compare import agreement, usable

    human, model = tmp_path / "human.jsonl", tmp_path / "m.jsonl"
    human.write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "stance": "bullish", "topic": "trading"},
        {"id": "2", "stance": "neutral", "topic": "chit_chat"},
        {"id": "3", "stance": None, "topic": None, "unsure": True},
        {"id": "4", "stance": "bearish", "topic": "macro"},
    ]), encoding="utf-8")
    model.write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "stance": "bullish", "topic": "trading"},
        {"id": "2", "stance": "neutral", "topic": "news_info"},
        {"id": "3", "stance": "bearish", "topic": "macro"},
        {"id": "4", "stance": "neutral", "topic": "macro"},
    ]), encoding="utf-8")

    r = agreement(usable(str(human)), usable(str(model)), "stance")
    assert r["n"] == 3 and r["accuracy"] == pytest.approx(2 / 3)
    assert r["confusion"]["labels"] == ["bearish", "bullish", "neutral"]
    assert r["confusion"]["matrix"] == [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
    assert agreement(usable(str(human)), {}, "stance") is None
