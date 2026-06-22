"""
Enrich messages with dialogue_act, sentiment, escalation markers.
Uses rule-based approach for Phase 1 (no LLM dependency).
"""
import re
from .parser import Message


# --- Dialogue Act ---

_QUESTION_PATTERNS = [
    r"[嗎呢？?]$",
    r"^是否",
    r"^可以.{0,10}[嗎？?]",
    r"^有沒有",
    r"^請問",
    r"^能不能",
    r"可以嗎",
    r"方便嗎",
    r"好嗎",
    r"對嗎",
    r"這樣可以",
]

_REQUEST_PATTERNS = [
    r"可以.{0,15}(給|提供|傳|發|做|改|幫)",
    r"(給|幫|請).{0,10}(我|一下)",
    r"(需要|想要|希望).{0,15}(可以|能)",
    r"麻煩",
    r"勞駕",
]

_ACK_PATTERNS = [
    r"^(好|ok|OK|收到|了解|知道了|沒問題|沒事|好的|好哦|嗯|恩|是|是的|收|ㄟ好)[!！。\s]*$",
    r"^(感謝|謝謝|感恩|謝)[!！。\s]*$",
    r"^好[，,]?(麻煩|辛苦|感謝|謝).{0,15}$",  # 「好，麻煩你們了！」類確認句
    r"👍",
]

_COMPLAINT_PATTERNS = [
    r"(太小|太大|不夠|不對|不好|不滿|有問題)",
    r"(覺得|感覺).{0,10}(不|沒)",
]

_CHIT_CHAT_PATTERNS = [
    r"^(早安|晚安|哈哈|😂|🥹|🙌|貼圖)[!！。\s]*$",
    r"颱風",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_dialogue_act(msg: Message) -> str:
    if msg.is_sticker or msg.is_image:
        return "ack"
    t = msg.text.strip()
    if not t:
        return "chit_chat"
    if _match_any(t, _ACK_PATTERNS):
        return "ack"
    if _match_any(t, _QUESTION_PATTERNS):
        return "question"
    if _match_any(t, _REQUEST_PATTERNS):
        return "request"
    if _match_any(t, _COMPLAINT_PATTERNS):
        return "report"
    if _match_any(t, _CHIT_CHAT_PATTERNS):
        return "chit_chat"
    return "answer"


# --- Sentiment ---

_NEGATIVE_WORDS = [
    # 情緒狀態
    "哭哭", "失望", "不滿", "不滿意", "不愉快", "不開心", "遺憾", "可惜",
    "生氣", "憤怒", "不耐煩", "抓狂", "崩潰", "心寒",
    # 品質/專業評價
    "不好", "太小", "太大", "太貴", "不夠", "不對", "不合理", "不專業",
    "太誇張", "離譜", "差勁", "有問題", "出問題", "品質差", "不符合",
    "無法接受", "難以接受", "不可以接受", "無法忍受",
    # 信任與承諾
    "難以相信", "不相信", "不信任", "信任度", "承諾跳票", "一再", "一直",
    "每次都", "說好", "說不定", "交代", "向董事會", "向上面",
    # 進度/延誤
    "嚴重落後", "進度落後", "一再延誤", "一再延期", "遲遲", "久等",
    # 服務態度
    "態度", "沒有回應", "都沒有人", "沒人理",
    # 升級意圖（也是負面訊號）
    "退費", "退款", "退訂", "投訴", "換人", "不合理",
    # 其他
    "困擾", "麻煩", "擔心", "疑慮", "拍謝", "抱歉", "對不起",
]
_POSITIVE_WORDS = [
    "感謝", "謝謝", "感恩", "好的", "沒問題", "完美", "很好", "讚", "棒",
    "期待", "開心", "順利", "👍", "🙌", "🥹",
]


def score_sentiment(text: str) -> float:
    """Return sentiment in [-1, 1]. Negative = bad."""
    if not text:
        return 0.0
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text)
    pos = sum(1 for w in _POSITIVE_WORDS if w in text)
    total = neg + pos
    if total == 0:
        return 0.0
    return (pos - neg) / total


# --- Escalation Markers ---

_ESCALATION_KEYWORDS = [
    "退費", "退錢", "退款", "退訂金", "退訂", "投訴", "換人",
    "找你主管", "找你老闆", "找主管", "找老闆", "跟你主管談", "跟你老闆談",
    "解約", "取消合約", "告你", "法院", "消保", "消費者保護",
]


def has_escalation_marker(text: str) -> bool:
    return any(kw in text for kw in _ESCALATION_KEYWORDS)


# --- Main enrichment entry point ---

def enrich(messages: list[Message]) -> list[Message]:
    for msg in messages:
        msg.dialogue_act = classify_dialogue_act(msg)
        msg.sentiment = score_sentiment(msg.text)
        msg.is_escalation_marker = has_escalation_marker(msg.text)
    return messages
