"""
Phase 2 — LLM Issue Extraction via Claude API.

Extracts structured issues from a window of LINE messages.
Only called when cheap metrics trigger or tripwire fires (tiered execution).
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic

from .parser import Message

_client: Optional[anthropic.Anthropic] = None

# LINE exports carry no timezone; timestamps are local time of the export (Taiwan)
CONVERSATION_TZ = timezone(timedelta(hours=8))


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


_SYSTEM = """\
你是 LINE 客服群組的對話分析助理。
你會收到一段對話紀錄（格式：[timestamp] [role] [sender]: [訊息]），
請辨識其中所有客戶提出的「議題」（問題、請求、投訴、回報）。

回傳嚴格符合以下 JSON Schema 的物件，不要加說明或 markdown：
{
  "issues": [
    {
      "issue_id": "string (短流水號, e.g. ISS-001)",
      "raised_by": "customer | staff",
      "raised_at": "YYYY-MM-DDTHH:MM:SS（沿用對話紀錄的本地時間，不轉時區、不加時區後綴）",
      "summary": "一句話（不超過40字）描述議題",
      "type": "question | request | complaint | report",
      "status": "resolved | unresolved | unclear",
      "resolution_evidence": "string（已解決時簡述依據；未解決則空字串）",
      "last_activity_at": "YYYY-MM-DDTHH:MM:SS（同上，本地時間）",
      "evidence_msg_ids": ["msg_id string"]
    }
  ]
}

判斷準則：
- 同一個客戶在短時間內連續追問同一件事，算一個議題。
- staff 有明確回答或確認完成即算 resolved。
- 判斷不清楚時填 unclear，不要猜。
- 如果整段對話沒有任何議題，回傳 {"issues": []}。
"""


def _build_conversation_text(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        ts = msg.timestamp.isoformat()
        role = msg.role or "unknown"
        sender = msg.sender or "?"
        text = "[貼圖]" if msg.is_sticker else (msg.text or "")
        lines.append(f"[{ts}] [{role}] [{msg.msg_id}] {sender}: {text}")
    return "\n".join(lines)


def extract_issues(
    group_id: str,
    messages: list[Message],
    model: str = "claude-haiku-4-5",
) -> list[dict]:
    """
    Call Claude to extract structured issues from messages.
    Returns list of issue dicts matching the spec schema.
    """
    if not messages:
        return []

    conv_text = _build_conversation_text(messages)
    client = _get_client()

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        thinking={"type": "disabled"},
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"以下是群組 [{group_id}] 的對話紀錄，請抽取所有議題：\n\n{conv_text}",
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
        issues = data.get("issues", [])
    except json.JSONDecodeError:
        # Try to extract JSON from response if there's surrounding text
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            issues = data.get("issues", [])
        else:
            issues = []

    # Attach group_id to each issue
    for iss in issues:
        iss["group_id"] = group_id

    return issues


def _to_local_naive(dt: datetime) -> datetime:
    """Normalize to naive local time so it compares with LINE message timestamps."""
    if dt.tzinfo is not None:
        return dt.astimezone(CONVERSATION_TZ).replace(tzinfo=None)
    return dt


def _issue_raised_at(iss: dict, msg_times: dict[str, datetime]) -> Optional[datetime]:
    """
    Prefer the earliest timestamp of the issue's evidence messages (ground truth);
    fall back to the LLM-reported raised_at.
    """
    evidence = [msg_times[mid] for mid in iss.get("evidence_msg_ids") or [] if mid in msg_times]
    if evidence:
        return min(evidence)
    try:
        return _to_local_naive(datetime.fromisoformat(iss.get("raised_at", "").replace("Z", "+00:00")))
    except (ValueError, TypeError, AttributeError):
        return None


def compute_i3(
    issues: list[dict],
    now: datetime,
    messages: Optional[list[Message]] = None,
    warn_min: float = 60,
    crit_min: float = 480,
) -> tuple[float, float]:
    """
    I3: Oldest Open Issue Age.
    Returns (severity, oldest_age_minutes).
    """
    if not issues:
        return 0.0, 0.0

    msg_times = {msg.msg_id: msg.timestamp for msg in messages or []}
    now = _to_local_naive(now)

    oldest_min = 0.0
    for iss in issues:
        if iss.get("status") not in ("unresolved", "unclear"):
            continue
        raised_at = _issue_raised_at(iss, msg_times)
        if raised_at is None:
            continue
        age_min = (now - raised_at).total_seconds() / 60
        if age_min > oldest_min:
            oldest_min = age_min

    if oldest_min <= 0:
        return 0.0, 0.0

    if oldest_min < warn_min:
        severity = 0.0
    elif oldest_min >= crit_min:
        severity = 1.0
    else:
        severity = (oldest_min - warn_min) / (crit_min - warn_min)

    return severity, oldest_min
