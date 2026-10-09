"""
Market entities mentioned in a message: stocks, ETFs, indices, sectors, assets.

A curated alias table instead of a generic regex: in these groups bare
4-digit numbers are mostly prices and years (1000, 2026), and upper-case
tokens are mostly XD / AI / QQ, so a pattern-based extractor drowns the real
tickers in noise. Each alias maps to one canonical entity id; an entity's
kind lets later steps look at stocks only, or themes only.

Only public names and codes live here — nothing taken from group messages.
"""
import re
from collections import Counter

# canonical id -> (kind, aliases). Codes / tickers are matched as whole tokens,
# Chinese names as substrings (longest alias first, so 台積電 wins over 積電).
ENTITIES: dict[str, tuple[str, list[str]]] = {
    # Taiwan stocks
    "2330": ("tw_stock", ["2330", "台積電", "台積", "積電", "台GG", "台G", "TSMC"]),
    "2454": ("tw_stock", ["2454", "聯發科", "發哥", "MTK"]),
    "2317": ("tw_stock", ["2317", "鴻海"]),
    "2303": ("tw_stock", ["2303", "聯電"]),
    "2344": ("tw_stock", ["2344", "華邦電", "華邦"]),
    "2408": ("tw_stock", ["2408", "南亞科"]),
    "2337": ("tw_stock", ["2337", "旺宏"]),
    "6770": ("tw_stock", ["6770", "力積電"]),
    "8299": ("tw_stock", ["8299", "群聯"]),
    "2327": ("tw_stock", ["2327", "國巨"]),
    "2308": ("tw_stock", ["2308", "台達電", "台達"]),
    "2301": ("tw_stock", ["2301", "光寶科", "光寶"]),
    "6669": ("tw_stock", ["6669", "緯穎"]),
    "2382": ("tw_stock", ["2382", "廣達"]),
    "3231": ("tw_stock", ["3231", "緯創"]),
    "2376": ("tw_stock", ["2376", "技嘉"]),
    "2603": ("tw_stock", ["2603", "長榮"]),
    "2609": ("tw_stock", ["2609", "陽明海運"]),
    "2615": ("tw_stock", ["2615", "萬海"]),
    "3008": ("tw_stock", ["3008", "大立光"]),
    "3711": ("tw_stock", ["3711", "日月光"]),
    "2368": ("tw_stock", ["2368", "金像電"]),
    "3661": ("tw_stock", ["3661", "世芯"]),
    "3443": ("tw_stock", ["3443", "創意電子"]),
    "5274": ("tw_stock", ["5274", "信驊"]),
    "3017": ("tw_stock", ["3017", "奇鋐"]),
    "2383": ("tw_stock", ["2383", "台光電"]),
    "3037": ("tw_stock", ["3037", "欣興"]),
    "2345": ("tw_stock", ["2345", "智邦"]),
    "3653": ("tw_stock", ["3653", "健策"]),
    "3081": ("tw_stock", ["3081", "聯亞"]),
    "2002": ("tw_stock", ["2002", "中鋼"]),
    "2881": ("tw_stock", ["2881", "富邦金"]),
    "2882": ("tw_stock", ["2882", "國泰金"]),
    # Taiwan ETFs
    "0050": ("tw_etf", ["0050"]),
    "0056": ("tw_etf", ["0056"]),
    "00878": ("tw_etf", ["00878"]),
    "00631L": ("tw_etf", ["00631L", "00631"]),
    "00692": ("tw_etf", ["00692"]),
    "00981A": ("tw_etf", ["00981A", "00981"]),
    # US stocks / ETFs
    "NVDA": ("us_stock", ["NVDA", "nvda", "Nvda", "輝達", "老黃"]),
    "TSM": ("us_stock", ["TSM"]),
    "TSLA": ("us_stock", ["TSLA", "tsla", "Tesla", "tesla", "特斯拉"]),
    "AAPL": ("us_stock", ["AAPL", "蘋果"]),
    "MU": ("us_stock", ["MU", "美光"]),
    "AVGO": ("us_stock", ["AVGO", "avgo", "博通"]),
    "AMD": ("us_stock", ["AMD", "超微"]),
    "GOOGL": ("us_stock", ["GOOGL", "GOOG", "谷歌"]),
    "MSFT": ("us_stock", ["MSFT", "微軟"]),
    "META": ("us_stock", ["META", "Meta"]),
    "AMZN": ("us_stock", ["AMZN", "亞馬遜"]),
    "PLTR": ("us_stock", ["PLTR", "pltr"]),
    "SNDK": ("us_stock", ["SNDK", "sndk"]),
    "MRVL": ("us_stock", ["MRVL"]),
    "QCOM": ("us_stock", ["QCOM", "qcom", "高通"]),
    "LITE": ("us_stock", ["LITE"]),
    "AXTI": ("us_stock", ["AXTI"]),
    "ON": ("us_stock", ["安森美"]),  # bare "ON" is an English word; only the Chinese name counts
    "QQQ": ("us_etf", ["QQQ", "qqq"]),
    "SOXL": ("us_etf", ["SOXL", "soxl"]),
    "SMH": ("us_etf", ["SMH"]),
    # indices
    "TAIEX": ("index", ["加權指數", "加權", "大盤", "台指期", "台指", "台股指數"]),
    "SOX": ("index", ["費半", "費城半導體", "SOX"]),
    "NASDAQ": ("index", ["那斯達克", "納斯達克", "納指", "NASDAQ"]),
    "SPX": ("index", ["標普", "S&P", "SPX", "SPY"]),
    "DJI": ("index", ["道瓊"]),
    # sectors / themes
    "memory": ("theme", ["記憶體", "DRAM", "HBM", "NAND"]),
    "semis": ("theme", ["半導體"]),
    "optical": ("theme", ["光通訊", "光通", "CPO", "矽光子"]),
    "passive": ("theme", ["被動元件", "MLCC"]),
    "cooling": ("theme", ["散熱"]),
    "pcb": ("theme", ["PCB", "ABF"]),
    "server": ("theme", ["伺服器"]),
    "robotics": ("theme", ["機器人"]),
    "shipping": ("theme", ["航運", "貨櫃"]),
    "defense": ("theme", ["軍工"]),
    "power": ("theme", ["重電"]),
    "biotech": ("theme", ["生技"]),
    "financials": ("theme", ["金融股", "金控"]),
    # other assets / macro
    "gold": ("asset", ["黃金"]),
    "silver": ("asset", ["白銀"]),
    "oil": ("asset", ["油價", "原油"]),
    "bitcoin": ("asset", ["比特幣", "BTC"]),
    "ust": ("asset", ["美債"]),
    "jpy": ("asset", ["日圓", "日幣"]),
}

KIND = {eid: kind for eid, (kind, _) in ENTITIES.items()}


def _is_token(alias: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9&]+", alias))


_alias_to_id = {a: eid for eid, (_, aliases) in ENTITIES.items() for a in aliases}
_aliases = sorted(_alias_to_id, key=len, reverse=True)
# ASCII aliases must stand alone (MU not inside MUST, 2330 not inside 23300);
# Chinese aliases match anywhere. One alternation, longest first, so overlaps resolve to the longer alias.
_PATTERN = re.compile("|".join(
    rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])" if _is_token(a) else re.escape(a) for a in _aliases
))


def extract_entities(text: str) -> list[str]:
    """Canonical entity ids in order of first mention, each once."""
    seen: dict[str, None] = {}
    for m in _PATTERN.finditer(text):
        seen.setdefault(_alias_to_id[m.group(0)], None)
    return list(seen)


def entity_counts(texts) -> Counter:
    """Messages mentioning each entity (a message counts once per entity)."""
    c: Counter = Counter()
    for t in texts:
        c.update(extract_entities(t))
    return c
