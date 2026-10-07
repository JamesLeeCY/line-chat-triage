"""
AI critique loop (RLAIF-style, no training): a judge model reviews another
labeller's stance against a written constitution and the label is revised.

The judge does not pick bullish / bearish / neutral directly. It answers the
constitution's checklist (is there a market target, what kind of message is
it, which direction does it point), and `derive_stance` applies the rules in
code. Small local models follow "answer these three questions" far better
than a long rule list — qwen3:8b broke prompt v3's own examples.

Output is an ordinary label source (labels/<base>+critic-<judge>.jsonl) with
the labeller's topic / tickers kept and a `critique` record per message, so
`compare` scores it against human labels like any other source.
"""
from typing import Literal

from pydantic import BaseModel, Field

from .gold import _format_batch
from .labelers import _slug

Kind = Literal[
    "view", "own_trade", "price_report", "market_news",                       # can carry stance
    "question", "wish_or_joke", "offtopic_slang", "pnl_only", "politics_social", "chit_chat",  # always neutral
]
NEUTRAL_KINDS = {"question", "wish_or_joke", "offtopic_slang", "pnl_only", "politics_social", "chit_chat"}

CONSTITUTION = """你是台灣股票社群訊息標註的審查員。另一位標註員已經為每則訊息標了多空立場（stance），請依下列憑法逐則檢查。
只根據「目標訊息」判斷；前文與被回覆的訊息只用來理解語意（例如「還在跌」是在說哪個標的）。

每則請回答：
1. critique：用一句話（30 字內）說明標註員的答案哪裡對或錯。
2. target：訊息談到的「市場標的」——個股、ETF、指數、期貨、選擇權、產業題材或整體股市。可從前文或被回覆的訊息推得。沒有就填空字串。
   股市用語用在非股市的事不算標的：被老闆炒、買房、訂位要搶、人生 GG、下車（下班、離開）。
3. kind：這則訊息屬於哪一類（只選一個）
   - view：對標的後續漲跌的看法；問句本身帶明顯看法也算（「是不是要跌回季線了」）
   - own_trade：自己的買進、加碼、續抱、回補空單、叫別人上車；或賣出、減碼、放空、停損、買 put 避險、被套
   - price_report：回報標的正在漲或已經漲 / 跌，包含只講漲跌幅或價位（「XX 漲停」「-7%」「又站回 1000」「還在跌」）
   - market_news：對標的明顯利多或利空的消息（漲價、需求強、砍單、財報不如預期）
   - question：單純提問，沒有表達自己看法（「年底會到多少？」「現在可以進嗎？」）
   - wish_or_joke：願望、假設、打賭、開玩笑的喊價（「希望明天跌一點」「要是漲 10% 我就請客」）
   - offtopic_slang：股市用語用在非股市的事
   - pnl_only：只說自己賺賠多少、資產多少，沒有指出哪個標的在漲跌
   - politics_social：政治、社會議題、八卦、時事評論
   - chit_chat：其他閒聊、純情緒、髒話、表情
4. direction：up（漲、買進、續抱、利多）、down（跌、賣出、放空、停損、被套、避險、利空）或 none。
   反諷依實際意思判斷（「好棒喔又跌停，明天繼續跌吧」→ down）。同一則提到不同方向時以訊息重點為準。

不要因為標註員已經給了答案就照抄；拿不準時 direction 填 none。每則輸入訊息都必須回傳一筆，id 原樣照抄。"""


class Critique(BaseModel):
    id: str = Field(description="The message id exactly as given")
    critique: str = Field(description="一句話說明標註員的答案哪裡對或錯")
    target: str = Field(description="市場標的；沒有就空字串")
    kind: Kind
    direction: Literal["up", "down", "none"]


class CritiqueBatch(BaseModel):
    critiques: list[Critique]


def derive_stance(c: Critique) -> str:
    """The constitution's rules, applied in code to the judge's checklist answers."""
    if not c.target.strip() or c.kind in NEUTRAL_KINDS or c.direction == "none":
        return "neutral"
    return "bullish" if c.direction == "up" else "bearish"


def _format_review(records: list[dict], base: dict[str, dict]) -> str:
    header = "請標註以下訊息：\n\n"
    parts = [_format_batch([r])[len(header):] + f"\n標註員的答案：stance={base[r['id']]['stance']}" for r in records]
    return "請審查以下訊息的標註：\n\n" + "\n\n".join(parts)


class Critic:
    """
    Wraps a judge backend (anything with `.ask(system, user, schema)` and
    `.model`, i.e. OllamaLabeler / ClaudeLabeler) and a base label source;
    exposes `.label(records)` so `gold.label_gold` drives it (batching,
    resume, retries) exactly like a labeller.
    """

    def __init__(self, judge, base_labels: dict[str, dict], base_name: str):
        self.judge, self.base = judge, base_labels
        self.model = judge.model
        self.name = f"{base_name}+critic-{_slug(judge.model)}"

    def label(self, records: list[dict]) -> list[dict]:
        records = [r for r in records if r["id"] in self.base]  # nothing to review without a base label
        if not records:
            return []
        result = self.judge.ask(CONSTITUTION, _format_review(records, self.base), CritiqueBatch)
        wanted, out = {r["id"] for r in records}, []
        for c in result.critiques:
            if c.id not in wanted:
                continue
            wanted.discard(c.id)
            base = self.base[c.id]
            stance = derive_stance(c)
            out.append({
                "id": c.id, "stance": stance, "topic": base["topic"], "tickers": base.get("tickers", []),
                "critique": {**c.model_dump(exclude={"id"}), "base_stance": base["stance"],
                             "changed": stance != base["stance"]},
            })
        return out
