import json

import pytest

from src.community.critique import Critic, Critique, CritiqueBatch, _format_review, derive_stance


def _c(id="a", target="台積電", kind="price_report", direction="up"):
    return Critique(id=id, critique="", target=target, kind=kind, direction=direction)


@pytest.mark.parametrize("kw,expected", [
    ({}, "bullish"),
    ({"direction": "down"}, "bearish"),
    ({"kind": "own_trade", "direction": "down"}, "bearish"),
    ({"target": ""}, "neutral"),                       # no market target → neutral
    ({"target": "  "}, "neutral"),
    ({"kind": "question"}, "neutral"),                 # neutral kinds override direction
    ({"kind": "wish_or_joke", "direction": "down"}, "neutral"),
    ({"kind": "offtopic_slang"}, "neutral"),
    ({"direction": "none"}, "neutral"),
])
def test_derive_stance_applies_constitution(kw, expected):
    assert derive_stance(_c(**kw)) == expected


class FakeJudge:
    model = "phi4:latest"

    def __init__(self, critiques):
        self.critiques, self.calls = critiques, []

    def ask(self, system, user, schema):
        assert schema is CritiqueBatch
        self.calls.append(user)
        return CritiqueBatch(critiques=self.critiques)


def test_critic_revises_and_keeps_base_topic():
    base = {"a": {"id": "a", "stance": "bearish", "topic": "individual_stock", "tickers": ["2330"]},
            "b": {"id": "b", "stance": "bullish", "topic": "chit_chat", "tickers": []}}
    judge = FakeJudge([_c("a"), _c("b", target="", kind="offtopic_slang"), _c("zzz")])  # zzz: not requested
    critic = Critic(judge, base, "qwen3-8b-v3")
    assert critic.name == "qwen3-8b-v3+critic-phi4-latest"

    out = critic.label([{"id": "a", "text": "漲停了"}, {"id": "b", "text": "被炒爽"}, {"id": "c", "text": "無底標"}])
    assert [r["id"] for r in out] == ["a", "b"]
    a, b = out
    assert (a["stance"], a["topic"], a["tickers"]) == ("bullish", "individual_stock", ["2330"])
    assert a["critique"]["changed"] and a["critique"]["base_stance"] == "bearish"
    assert b["stance"] == "neutral" and b["critique"]["kind"] == "offtopic_slang"
    # the judge saw each message with the labeller's answer, but not message c (no base label)
    assert "漲停了\n標註員的答案：stance=bearish" in judge.calls[0] and "無底標" not in judge.calls[0]
    json.dumps(out, ensure_ascii=False)


def test_critic_skips_call_without_base_labels():
    judge = FakeJudge([])
    assert Critic(judge, {}, "x").label([{"id": "a", "text": "t"}]) == [] and judge.calls == []


def test_format_review_marks_each_target():
    text = _format_review([{"id": "a", "text": "噴"}, {"id": "b", "text": "噴", "previous": ["前"]}],
                          {"a": {"stance": "bullish"}, "b": {"stance": "neutral"}})
    assert text.startswith("請審查以下訊息的標註")
    a, b = text.split("\n\n")[1:]
    assert a.endswith("目標訊息：噴\n標註員的答案：stance=bullish")    # identical texts stay paired with their own answer
    assert b.startswith("### id: b") and b.endswith("目標訊息：噴\n標註員的答案：stance=neutral")
