"""
Scenario scripts for synthetic manager-triage groups, with ground truth.

Code — not the generating model — decides every message's day, time, sender
and intent, and which risks are planted. The model (generate.py) only puts
those intents into words. A small local model cannot be trusted to keep
"no staff reply for six business hours" true, but the skeleton can, so the
ground truth here holds by construction. What still needs checking is whether
the wording carries each intent (a threat that reads as a threat), which is
the validator's job (validate.py).

Ground truth describes the group's state at `now` (end of the last day):
- unanswered_question  a customer question with no staff reply since (≥ 6 business hours)
- slow_response        staff routinely answer hours late
- escalation           customer threatens refund / complaint / manager / cancellation within 72 h
- negative_sentiment   customer increasingly frustrated, without an explicit threat
Decoys look risky but are not, to measure false alarms:
- resolved_escalation  a threat more than 72 h ago that staff resolved, customer satisfied since
- after_hours          a late-night question answered first thing next morning
"""
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta

ALERTS = ("unanswered_question", "slow_response", "escalation", "negative_sentiment")
DECOYS = ("resolved_escalation", "after_hours")
HIGH_ALERTS = {"unanswered_question", "escalation"}

INDUSTRIES = [
    ("住宅翻新", ["防水工程", "磁磚", "廚具", "水電配管", "油漆"]),
    ("餐廳設計", ["吧台", "廚房動線", "燈具", "招牌", "座椅"]),
    ("牙醫診所", ["診療椅配管", "候診區", "X 光室屏蔽", "櫃台", "消防"]),
    ("健身房規劃", ["地板", "器材進場", "淋浴間", "空調", "鏡面牆"]),
    ("品牌旗艦店", ["展示櫃", "櫥窗", "燈光", "收銀台", "試衣間"]),
    ("民宿改裝", ["客房衛浴", "外牆", "庭院", "床組", "熱水系統"]),
    ("辦公室規劃", ["隔間", "網路佈線", "會議室", "茶水間", "地毯"]),
    ("服飾選品店", ["衣架系統", "倉儲", "試衣間", "照明", "門面"]),
    ("咖啡廳裝潢", ["咖啡機配電", "吧台", "戶外座位", "排風", "木作"]),
    ("補習班教室", ["教室隔音", "投影設備", "櫃台", "冷氣", "逃生動線"]),
]
SURNAMES = list("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾")
TITLES = ["先生", "小姐", "太太", "老闆", "經理", "醫師", "總"]
PERSONAS = ["客氣有禮", "講求效率、說話直接", "細節多、常追問", "忙碌、回覆簡短"]
STAFF = ["成員1", "成員2", "成員3", "成員4"]
WEEKDAYS = "一二三四五六日"

N_DAYS = 4                       # short conversations: 4 days, ~25–35 messages
NOW_TIME = time(18, 30)          # ground truth is the state at 18:30 on the last day


@dataclass
class Slot:
    day: int                      # 0 .. N_DAYS-1
    time: str                     # HH:MM
    role: str                     # customer | staff
    speaker: str
    intent: str                   # what the message must say (Chinese, for the generator)
    tag: str = ""                 # planted risk / decoy this message carries, if any


@dataclass
class Scenario:
    group: str
    industry: str
    customer: str
    persona: str
    topics: list
    start: str                    # ISO date of day 0
    now: str                      # ISO datetime the truth refers to
    risk_level: str               # low | medium | high
    alerts: list
    decoys: list
    slots: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["slots"] = [asdict(s) for s in self.slots]
        return d


def _t(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def _risk_level(alerts: list) -> str:
    if HIGH_ALERTS & set(alerts) or len(alerts) >= 2:
        return "high"
    return "medium" if alerts else "low"


def _pick_risks(rng: random.Random) -> tuple[list, list]:
    """About 30% healthy; otherwise one or two planted risks. Decoys independently ~35%."""
    if rng.random() < 0.3:
        alerts = []
    else:
        k = 1 if rng.random() < 0.65 else 2
        alerts = rng.sample(ALERTS, k)
    decoys = [d for d in DECOYS if rng.random() < 0.35]
    if "escalation" in alerts and "resolved_escalation" in decoys:
        decoys.remove("resolved_escalation")      # keep the two escalation stories apart
    return alerts, decoys


def build_scenario(i: int, rng: random.Random) -> Scenario:
    industry, topics = rng.choice(INDUSTRIES)
    customer = f"客戶{chr(65 + i % 26)}{rng.choice(SURNAMES)}{rng.choice(TITLES)}"
    staff = rng.sample(STAFF, 2)
    alerts, decoys = _pick_risks(rng)
    # all four days on weekdays (Mon–Thu or Tue–Fri): service hours are Mon–Fri 09:00–18:00, so a
    # weekend would silently stop the "unanswered for 6 business hours" clock
    start = date(2026, 3, 2) + timedelta(weeks=rng.randrange(0, 26), days=rng.choice([0, 1]))
    topics = rng.sample(topics, 3)
    slow = "slow_response" in alerts
    slots: list[Slot] = []

    def exchange(day: int, h: int, topic: str, kind: str = "question", tag: str = ""):
        """Customer asks / requests at h:mm; staff answer after a normal or slow gap; customer acks."""
        m = rng.randrange(0, 50)
        intent = {"question": f"（我是客戶）問員工「{topic}」現在的進度或細節",
                  "request": f"（我是客戶）請員工修改「{topic}」的某個地方"}[kind]
        slots.append(Slot(day, _t(h, m), "customer", customer, intent, tag))
        gap = rng.randrange(180, 300) if slow else rng.randrange(5, 40)
        reply = datetime(2000, 1, 1, h, m) + timedelta(minutes=gap)
        if reply.hour >= 18:          # keep slow replies inside the working day
            reply = reply.replace(hour=17, minute=rng.randrange(30, 59))
        slots.append(Slot(day, reply.strftime("%H:%M"), "staff", rng.choice(staff),
                          f"（我是員工）回答客戶關於「{topic}」的問題，給出具體說明或時間" + ("（回得很晚，可順口致歉）" if slow else ""),
                          "slow_response" if slow else ""))
        if rng.random() < 0.6:
            ack = reply + timedelta(minutes=rng.randrange(2, 20))
            if ack.hour < 18:
                slots.append(Slot(day, ack.strftime("%H:%M"), "customer", customer, "（我是客戶）簡短回應：好、收到或謝謝"))

    last = N_DAYS - 1
    for day in range(N_DAYS):
        slots.append(Slot(day, _t(9, rng.randrange(0, 30)), "staff", rng.choice(staff),
                          f"（我是員工）主動跟客戶回報今天的施工 / 工作安排（{topics[day % 3]}）"))
        exchange(day, rng.choice([10, 11]), topics[day % 3], rng.choice(["question", "request"]))
        if not (day == last and "unanswered_question" in alerts):
            exchange(day, rng.choice([14, 15]), topics[(day + 1) % 3])

    if "resolved_escalation" in decoys:
        # day 0, before 10:00 → more than 72 h before now; resolved the same day. The routine
        # morning update would land mid-argument, so day 0 starts with the complaint instead.
        slots = [s for s in slots if not (s.day == 0 and "主動跟客戶回報" in s.intent)]
        slots += [
            Slot(0, _t(8, 40), "customer", customer, "（我是客戶）對你們的某個疏失非常不滿，直接說要退訂金，不然就要找你們主管", "resolved_escalation"),
            Slot(0, _t(9, 5), "staff", staff[0], "（我是員工）向客戶誠懇道歉，提出具體補救方案與時間", "resolved_escalation"),
            Slot(0, _t(9, 20), "customer", customer, "（我是客戶）接受補救方案，語氣緩和下來", "resolved_escalation"),
            Slot(1, _t(16, 40), "customer", customer, "（我是客戶）說補救後的成果我很滿意，謝謝你們", "resolved_escalation"),
        ]
    if "after_hours" in decoys:
        day = rng.randrange(0, last)
        slots += [
            Slot(day, _t(22, rng.randrange(10, 50)), "customer", customer,
                 f"（我是客戶）深夜想到「{topics[2]}」的一個問題，順手問一下，不急、語氣平和", "after_hours"),
            Slot(day + 1, _t(8, rng.randrange(40, 59)), "staff", staff[1],
                 f"（我是員工）一早回答客戶昨晚問的「{topics[2]}」問題", "after_hours"),
        ]
    if "negative_sentiment" in alerts:
        for day, h in ((last - 1, 16), (last, 12), (last, 17)):
            slots.append(Slot(day, _t(h, rng.randrange(0, 50)), "customer", customer,
                              "（我是客戶）抱怨你們的品質、進度或溝通，很失望、語氣越來越差，但不提退款、投訴或找主管",
                              "negative_sentiment"))
    if "escalation" in alerts:
        day = rng.choice([last - 1, last])
        slots.append(Slot(day, _t(16, rng.randrange(0, 50)), "customer", customer,
                          "（我是客戶）非常生氣，直接揚言要退款、投訴、解約或找你們主管", "escalation"))
        slots.append(Slot(day, _t(17, rng.randrange(0, 50)), "staff", staff[0],
                          "（我是員工）向客戶道歉並說會向上回報，但沒有提出具體解決方案"))
    if "unanswered_question" in alerts:
        # nothing from staff after this question until now (18:30): about 8 business hours
        slots = [s for s in slots if not (s.day == last and s.role == "staff" and s.time > "10:00")]
        question = [
            Slot(last, _t(10, rng.randrange(0, 20)), "customer", customer,
                 f"（我是客戶）問員工一個關於「{topics[0]}」、需要他們回答的具體問題", "unanswered_question"),
            Slot(last, _t(14, rng.randrange(0, 30)), "customer", customer,
                 "（我是客戶）追問剛才的問題，問有沒有人看到", "unanswered_question"),
        ]
        # with the staff replies gone, a later "好，收到" from the customer thanks nobody and reads as
        # if the question were answered; drop those acks too (no rng calls here, so every other
        # scenario stays identical)
        slots = [s for s in slots if not (s.day == last and s.role == "customer" and "簡短回應" in s.intent
                                          and s.time > question[0].time)] + question

    slots.sort(key=lambda s: (s.day, s.time))
    now = datetime.combine(start + timedelta(days=last), NOW_TIME)
    return Scenario(
        group=f"{industry}_{customer[3:]}_{i:03d}", industry=industry, customer=customer,
        persona=rng.choice(PERSONAS), topics=topics, start=start.isoformat(), now=now.isoformat(timespec="minutes"),
        risk_level=_risk_level(alerts), alerts=sorted(alerts), decoys=sorted(decoys), slots=slots,
    )


def build_scenarios(n: int, seed: int = 0) -> list[Scenario]:
    rng = random.Random(seed)
    return [build_scenario(i, rng) for i in range(n)]


def day_header(start: str, day: int) -> str:
    d = date.fromisoformat(start) + timedelta(days=day)
    return f"{d:%Y.%m.%d} 星期{WEEKDAYS[d.weekday()]}"
