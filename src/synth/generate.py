"""
Model A: put a scenario's message intents into words, then render a LINE
export (.txt) that the existing parser reads unchanged.

One call per group. The model sees the story context and the numbered slots
(time, sender, intent) and returns one text per slot; times and senders come
from the scenario, so the model cannot move a reply or drop a planted gap.
Ground truth (scenario + risks) is written next to the conversations.
Resumable: groups whose .txt already exists are skipped.
"""
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .scenarios import Scenario, day_header

SYSTEM = """你是對話編劇，替台灣的設計 / 工程公司產生逼真的 LINE 工作群組訊息。群組裡只有兩種人：
- 客戶本人（業主）：付錢的人，對公司員工說話。
- 公司員工（「成員」開頭）：專案人員，對客戶說話，可以稱呼客戶（例如「郭先生」）。

你會拿到依時間排好的訊息清單；每一則都標明【客戶說】或【員工說】，以及這則要表達的意思（用「我是客戶 / 我是員工」的角度寫）。請為每一則寫出那個人實際會打的訊息：

- 【客戶說】：用客戶自己的口吻、第一人稱。客戶不會叫自己的名字，不會說「我們會改善」「向上回報」「請再給我們一點時間」這類公司的話，也不會替公司道歉。
- 【員工說】：用員工的口吻對客戶說話。
- 繁體中文、台灣口語，像真的 LINE 訊息：簡短（多半 5–40 字），可以有語助詞、表情符號，不要寫成書信。
- 每則都要確實表達指定的意思：要生氣就真的生氣，要揚言就把要求直接講出來；前後連貫，細節（日期、材料）一致，可自行編造合理的專案細節。
- 只有在指定意思要求時才寫出退款、投訴、解約、找主管等字眼；其他訊息不要出現這些字。
- 不要提到真實存在的公司或品牌名稱。

範例（只示範口吻，內容請自己寫）：
【客戶說】（我是客戶）問員工某個進度 → 「請問浴室的磁磚這週會到嗎？」
【客戶說】（我是客戶）很生氣、揚言 → 「已經延了兩次了，再這樣我要退訂金，你們主管是誰？」
【員工說】（我是員工）道歉並說明 → 「王先生不好意思，師傅臨時改期，明天上午一定到場。」

每一則都要回傳，序號 i 原樣照抄。"""


class Line(BaseModel):
    i: int = Field(description="Slot number exactly as given")
    text: str


class Lines(BaseModel):
    messages: list[Line]


def _prompt(sc: Scenario) -> str:
    head = (f"專案：{sc.industry}。客戶是「{sc.customer[3:]}」（群組顯示名稱「{sc.customer}」），個性：{sc.persona}。"
            f"本週會談到的項目：{'、'.join(sc.topics)}。\n\n訊息清單：")
    rows = [f"{i}. 第{s.day + 1}天 {s.time}【{'客戶說' if s.role == 'customer' else '員工說'}】{s.intent}"
            for i, s in enumerate(sc.slots)]
    return head + "\n" + "\n".join(rows)


# Phrases that only make sense from the company side; in a customer's mouth they mean the
# model swapped roles (the pilot's main failure: "threats" written as staff apologies).
_STAFF_VOICE = ("我們會", "我們已經", "我們將", "向上回報", "向上反映", "給我們一點時間", "給我們一次機會",
                "請您放心", "造成您", "我們確實", "我們這邊")


def role_flips(sc: Scenario, texts: list[str]) -> list[int]:
    """
    Customer slots written in the staff's voice: the customer addressing themselves by
    name, or company-side phrases. Deliberately checks voice only — whether a threat
    reads as a threat is left to the validator, since a keyword check here would mirror
    the rule system's tripwire list and inflate its score.
    """
    own_name = sc.customer[3:]                     # 郭先生 from 客戶A郭先生
    return [i for i, (s, t) in enumerate(zip(sc.slots, texts))
            if s.role == "customer" and (own_name in t or any(p in t for p in _STAFF_VOICE))]


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


def write_text(sc: Scenario, model, attempts: int = 3) -> tuple[list[str], dict]:
    """
    Texts for every slot, in order, plus a QA record. Rewrites the whole group when
    slots come back empty or customer messages are in the staff's voice; after
    `attempts`, keeps the version with the fewest role flips and records them.
    """
    wanted = len(sc.slots)
    best, best_flips, missing = None, None, None
    for attempt in range(1, attempts + 1):
        result = model.ask(SYSTEM, _prompt(sc), Lines)
        by_i = {m.i: m.text.strip() for m in result.messages if 0 <= m.i < wanted and m.text.strip()}
        missing = [i for i in range(wanted) if i not in by_i]
        if missing:
            continue
        texts = [by_i[i] for i in range(wanted)]
        flips = role_flips(sc, texts)
        if best is None or len(flips) < len(best_flips):
            best, best_flips = texts, flips
        if not flips:
            break
    if best is None:
        raise RuntimeError(f"{sc.group}: 第 {missing} 則沒有內容")
    return best, {"attempts": attempt, "role_flips": best_flips}


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
        try:
            texts, qa = write_text(sc, model)
        except Exception as e:                 # keep going; a rerun retries the failed groups
            stats["failed"] += 1
            print(f"[generate] {sc.group} 失敗：{e}", file=sys.stderr)
            continue
        (conv / f"{sc.group}.txt").write_text(render_line_export(sc, texts), encoding="utf-8")
        with open(truth_path, "a", encoding="utf-8") as f:
            # qa: generation checks, not ground truth; role_flips = customer slots still in the
            # staff's voice after the retries
            f.write(json.dumps({**sc.as_dict(), "qa": qa}, ensure_ascii=False) + "\n")
        stats["generated"] += 1
        elapsed = time.time() - started
        flag = f"，仍有 {len(qa['role_flips'])} 則角色錯亂" if qa["role_flips"] else ""
        print(f"[generate] {k}/{len(todo)} {sc.group}（{len(sc.slots)} 則，第 {qa['attempts']} 次{flag}）已花 {elapsed / 60:.0f} 分，"
              f"預估剩 {elapsed / k * (len(todo) - k) / 60:.0f} 分", file=sys.stderr)
    return stats
