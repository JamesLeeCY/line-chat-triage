"""
Model A: put a scenario's message intents into words, then render a LINE
export (.txt) that the existing parser reads unchanged.

One call per day (6–9 messages), with the days already written as context.
In the second pilot a single 27-message call drifted out of alignment about
a third of the way in — the dialogue stayed fluent but texts slid onto the
wrong slots, so customer slots got staff apologies and vice versa. Short
calls keep a small model aligned, and each message must echo who is speaking
(`who`), which forces a per-slot check and makes any slip detectable.

Times and senders come from the scenario, so the model cannot move a reply
or drop a planted gap. A day that comes back misaligned or in the wrong voice
is rewritten (up to `attempts` times); the best version is kept and flagged.
Ground truth (scenario + risks) is written next to the conversations.
Resumable: groups already in truth.jsonl are skipped.
"""
import json
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .progress import call_stats, log_progress
from .scenarios import Scenario, day_header

SYSTEM = """你是對話編劇，替台灣的設計 / 工程公司產生逼真的 LINE 工作群組訊息。群組裡只有兩種人：
- 客戶本人（業主）：付錢的人，對公司員工說話。
- 公司員工（「成員」開頭）：專案人員，對客戶說話，可以稱呼客戶（例如「郭先生」）。

你會拿到前幾天已經發生的對話，以及「今天」依時間排好的訊息清單；每一則都標明【客戶說】或【員工說】，以及這則要表達的意思（用「我是客戶 / 我是員工」的角度寫）。請只為今天的每一則寫出那個人實際會打的訊息：

- 每一則都要回傳序號 i（原樣照抄）、who（照抄標記：客戶或員工）和 text。先看清楚 who 是誰，再用那個人的口吻寫。
- 客戶：用客戶自己的口吻、第一人稱。客戶不會叫自己的名字，不會說「我們會改善」「向上回報」「請再給我們一點時間」這類公司的話，也不會替公司道歉。
- 員工：用員工的口吻對客戶說話。
- 繁體中文、台灣口語，像真的 LINE 訊息：簡短（多半 5–40 字），可以有語助詞、表情符號，不要寫成書信。
- 每則都要確實表達指定的意思：要生氣就真的生氣，要揚言就把要求直接講出來；和前幾天的對話連貫，細節（日期、材料）一致，可自行編造合理的專案細節。
- 只有在指定意思要求時才寫出退款、投訴、解約、找主管等字眼；其他訊息不要出現這些字。
- 不要提到真實存在的公司或品牌名稱。

範例（只示範格式與口吻，內容請自己寫）：
3.【客戶說】（我是客戶）問員工某個進度 → {"i": 3, "who": "客戶", "text": "請問浴室的磁磚這週會到嗎？"}
4.【員工說】（我是員工）回答並說明 → {"i": 4, "who": "員工", "text": "王先生您好，磁磚週四下午到，到了馬上跟您說～"}
5.【客戶說】（我是客戶）很生氣、揚言 → {"i": 5, "who": "客戶", "text": "已經延了兩次了，再這樣我要退訂金，你們主管是誰？"}"""

WHO = {"customer": "客戶", "staff": "員工"}


class Line(BaseModel):
    i: int = Field(description="Slot number exactly as given")
    who: Literal["客戶", "員工"] = Field(description="照抄這則的標記：客戶或員工")
    text: str


class Lines(BaseModel):
    messages: list[Line]


def _prompt_day(sc: Scenario, day: int, written: dict[int, str]) -> str:
    head = (f"專案：{sc.industry}。客戶是「{sc.customer[3:]}」（群組顯示名稱「{sc.customer}」），個性：{sc.persona}。"
            f"本週會談到的項目：{'、'.join(sc.topics)}。")
    before = [f"第{s.day + 1}天 {s.time} {WHO[s.role]}：{written[i]}" for i, s in enumerate(sc.slots)
              if s.day < day and i in written]
    today = [f"{i}.【{WHO[s.role]}說】{s.time}（{s.speaker}）{s.intent}" for i, s in enumerate(sc.slots) if s.day == day]
    parts = [head]
    if before:
        parts.append("前幾天已經發生的對話：\n" + "\n".join(before))
    parts.append(f"今天是第{day + 1}天，請寫以下訊息：\n" + "\n".join(today))
    return "\n\n".join(parts)


# Phrases that only make sense from the company side; in a customer's mouth they mean the
# model swapped roles (pilot 1's main failure: "threats" written as staff apologies) ...
_STAFF_VOICE = ("我們會", "我們已經", "我們將", "向上回報", "向上反映", "給我們一點時間", "給我們一次機會",
                "請您放心", "造成您", "我們確實", "我們這邊")
# ... and the customer side's planning / demands, which pilot 2 found in staff slots
_CUSTOMER_VOICE = ("我這邊要安排", "我這邊要", "請你們", "你們什麼時候", "我比較喜歡", "找你們主管", "我要退")


def role_flips(sc: Scenario, texts: list[str]) -> list[int]:
    """
    Slots written in the other side's voice: a customer addressing themselves by name or
    using company phrases, or staff talking like the customer. Deliberately checks voice
    only — whether a threat reads as a threat is left to the validator, since a keyword
    check here would mirror the rule system's tripwire list and inflate its score.
    """
    own_name = sc.customer[3:]                     # 郭先生 from 客戶A郭先生
    flipped = []
    for i, (s, t) in enumerate(zip(sc.slots, texts)):
        if t is None:
            continue
        if s.role == "customer" and (own_name in t or any(p in t for p in _STAFF_VOICE)):
            flipped.append(i)
        elif s.role == "staff" and any(p in t for p in _CUSTOMER_VOICE):
            flipped.append(i)
    return flipped


def render_line_export(sc: Scenario, texts: list[str]) -> str:
    """LINE desktop export format, as parsed by src.parser.parse_file."""
    out = [f"[LINE] {sc.group}", f"儲存日期：{sc.now[:10].replace('-', '/')} {sc.now[11:]}", ""]
    day = None
    for s, text in zip(sc.slots, texts):
        if s.day != day:
            if day is not None:
                out.append("")
            out.append(day_header(sc.start, s.day))
            day = s.day
        clean = " ".join(text.split())          # one LINE row per message
        out.append(f"{s.time}\t{s.speaker}\t{clean}")
    return "\n".join(out) + "\n"


class GenerationFailed(RuntimeError):
    def __init__(self, message: str, attempt_log: list):
        super().__init__(message)
        self.attempt_log = attempt_log


def _write_day(sc: Scenario, day: int, written: dict[int, str], model, attempts: int, log: list) -> list[int]:
    """Fill `written` for one day's slots; returns the slots still flagged after the retries."""
    slots = [i for i, s in enumerate(sc.slots) if s.day == day]
    best, best_bad = None, None
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        result = model.ask(SYSTEM, _prompt_day(sc, day, written), Lines)
        got = {m.i: m for m in result.messages if m.i in slots and m.text.strip()}
        missing = [i for i in slots if i not in got]
        texts = [got[i].text.strip() if i in got else None for i in range(len(sc.slots))]
        echo = [i for i in got if got[i].who != WHO[sc.slots[i].role]]      # model lost track of who speaks
        voice = [i for i in role_flips(sc, texts) if i in got]
        bad = sorted(set(echo) | set(voice))
        log.append({"day": day, "attempt": attempt, **call_stats(model, t0),
                    "missing": len(missing), "who_mismatch": len(echo), "role_flips": len(voice)})
        if missing:
            continue
        if best is None or len(bad) < len(best_bad):
            best, best_bad = {i: got[i].text.strip() for i in slots}, bad
        if not bad:
            break
    if best is None:
        raise GenerationFailed(f"{sc.group}: 第{day + 1}天的第 {missing} 則沒有內容", log)
    written.update(best)
    return best_bad


def write_text(sc: Scenario, model, attempts: int = 3) -> tuple[list[str], dict]:
    """
    Texts for every slot, in order, plus a QA record: total model calls and the
    slots still misaligned or in the wrong voice after the per-day retries.
    """
    written: dict[int, str] = {}
    log: list = []
    flagged: list[int] = []
    for day in sorted({s.day for s in sc.slots}):
        flagged += _write_day(sc, day, written, model, attempts, log)
    texts = [written[i] for i in range(len(sc.slots))]
    return texts, {"attempts": len(log), "role_flips": flagged, "attempt_log": log}


def generate(scenarios: list[Scenario], model, out_dir: str) -> dict:
    out = Path(out_dir)
    conv = out / "conversations"
    conv.mkdir(parents=True, exist_ok=True)
    (out / "employees.txt").write_text("# 合成群組的員工帳號\n成員1\n成員2\n成員3\n成員4\n", encoding="utf-8")
    truth_path = out / "truth.jsonl"
    done = {json.loads(line)["group"] for line in truth_path.read_text(encoding="utf-8").splitlines()} \
        if truth_path.exists() else set()
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    started = time.time()
    todo = [sc for sc in scenarios if sc.group not in done]
    stats["skipped"] = len(scenarios) - len(todo)
    for k, sc in enumerate(todo, 1):
        t0 = time.time()
        try:
            texts, qa = write_text(sc, model)
        except Exception as e:                 # keep going; a rerun retries the failed groups
            stats["failed"] += 1
            log_progress(out, "generate", sc.group, status="failed", seconds=round(time.time() - t0, 1),
                         error=str(e), attempt_log=getattr(e, "attempt_log", None))
            print(f"[generate] {sc.group} 失敗：{e}", file=sys.stderr)
            continue
        attempt_log = qa.pop("attempt_log")
        (conv / f"{sc.group}.txt").write_text(render_line_export(sc, texts), encoding="utf-8")
        with open(truth_path, "a", encoding="utf-8") as f:
            # qa: generation checks, not ground truth; role_flips = slots still misaligned or in
            # the wrong voice after the retries
            f.write(json.dumps({**sc.as_dict(), "qa": qa}, ensure_ascii=False) + "\n")
        log_progress(out, "generate", sc.group, status="ok", seconds=round(time.time() - t0, 1),
                     attempts=qa["attempts"], role_flips=len(qa["role_flips"]), n_messages=len(sc.slots),
                     attempt_log=attempt_log)
        stats["generated"] += 1
        elapsed = time.time() - started
        flag = f"，仍有 {len(qa['role_flips'])} 則錯位或口吻不符" if qa["role_flips"] else ""
        print(f"[generate] {k}/{len(todo)} {sc.group}（{len(sc.slots)} 則，呼叫 {qa['attempts']} 次{flag}）"
              f"已花 {elapsed / 60:.0f} 分，預估剩 {elapsed / k * (len(todo) - k) / 60:.0f} 分", file=sys.stderr)
    return stats
