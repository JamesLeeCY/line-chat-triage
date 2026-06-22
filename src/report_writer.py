"""
PDF report generator for LINE group triage metrics.
Uses reportlab — no system dependencies required.
"""
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, PolyLine, Line, Rect, String as GString

from .metrics import GroupMetrics

# --- Font setup (fallback to built-in if no CJK font found) ---
_CJK_FONT = "Helvetica"
_CJK_BOLD = "Helvetica-Bold"

def _register_cjk_font():
    global _CJK_FONT, _CJK_BOLD
    candidates = [
        # Windows
        ("C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc"),
        ("C:/Windows/Fonts/mingliu.ttc", "C:/Windows/Fonts/mingliu.ttc"),
        ("C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simsunb.ttf"),
        # macOS
        ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc"),
        # Linux
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
         "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for reg, bold in candidates:
        if Path(reg).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJK", reg))
                pdfmetrics.registerFont(TTFont("CJK-Bold", bold))
                _CJK_FONT = "CJK"
                _CJK_BOLD = "CJK-Bold"
                return
            except Exception:
                continue

_register_cjk_font()

# --- Color palette ---
C_DARK    = colors.HexColor("#1a1a1a")
C_MUTED   = colors.HexColor("#6b7280")
C_BORDER  = colors.HexColor("#e5e7eb")
C_GREEN   = colors.HexColor("#3c8a4a")
C_GREEN_BG= colors.HexColor("#eaf3de")
C_RED     = colors.HexColor("#e24b4a")
C_RED_BG  = colors.HexColor("#fcebeb")
C_AMBER   = colors.HexColor("#ba7517")
C_AMBER_BG= colors.HexColor("#faeeda")
C_BLUE    = colors.HexColor("#185fa5")
C_BLUE_BG = colors.HexColor("#e6f1fb")
C_HEADER_BG = colors.HexColor("#f3f4f6")
C_WHITE   = colors.white

W = A4[0] - 28*mm  # usable width

# --- Paragraph styles ---
def _style(name, font=None, size=10, leading=None, color=C_DARK,
           align=TA_LEFT, space_before=0, space_after=4):
    return ParagraphStyle(
        name,
        fontName=font or _CJK_FONT,
        fontSize=size,
        leading=leading or size * 1.4,
        textColor=color,
        alignment=align,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )

S_TITLE     = _style("title",     size=18, font=_CJK_BOLD, space_after=2)
S_SUBTITLE  = _style("subtitle",  size=10, color=C_MUTED,  space_after=12)
S_H2        = _style("h2",        size=12, font=_CJK_BOLD, space_before=14, space_after=4)
S_BODY      = _style("body",      size=9,  leading=14,     space_after=3)
S_SMALL     = _style("small",     size=8,  color=C_MUTED,  space_after=2)
S_INSIGHT   = _style("insight",   size=9,  leading=14, color=C_MUTED, space_after=0)
S_BADGE_G   = _style("badge_g",   size=8,  font=_CJK_BOLD, color=C_GREEN,  align=TA_CENTER)
S_BADGE_R   = _style("badge_r",   size=8,  font=_CJK_BOLD, color=C_RED,    align=TA_CENTER)
S_BADGE_A   = _style("badge_a",   size=8,  font=_CJK_BOLD, color=C_AMBER,  align=TA_CENTER)
S_BADGE_B   = _style("badge_b",   size=8,  font=_CJK_BOLD, color=C_BLUE,   align=TA_CENTER)
S_RIGHT     = _style("right",     size=9,  align=TA_RIGHT)


def _severity_color(sev: float):
    if sev >= 0.7:
        return C_RED
    if sev >= 0.35:
        return C_AMBER
    return C_GREEN


def _bar_table(label: str, sev: float, display: str) -> Table:
    """Single horizontal bar row. Columns sum exactly to W."""
    bar_color = _severity_color(sev)
    COL_LABEL = W * 0.28
    COL_BAR   = W * 0.42
    COL_NUM   = W * 0.30   # wide enough for Chinese display text

    # Build two-segment progress bar as a nested table
    filled_w = sev * COL_BAR
    empty_w  = COL_BAR - filled_w
    if filled_w > 0 and empty_w > 0:
        inner = Table([[" ", " "]], colWidths=[filled_w, empty_w], rowHeights=[7])
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,0), bar_color),
            ("BACKGROUND",    (1,0),(1,0), C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))
    elif filled_w == 0:
        inner = Table([[" "]], colWidths=[COL_BAR], rowHeights=[7])
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,0), C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))
    else:
        inner = Table([[" "]], colWidths=[COL_BAR], rowHeights=[7])
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,0), bar_color),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))

    num_style = _style(f"rv_{label}", size=8, color=bar_color, align=TA_LEFT, space_after=0)
    row = Table(
        [[Paragraph(label, S_BODY), inner, Paragraph(display, num_style)]],
        colWidths=[COL_LABEL, COL_BAR, COL_NUM],
        rowHeights=[18],
    )
    row.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (2,0),(2,0),   6),
    ]))
    return row


def _metric_cards(metrics: list[tuple]) -> Table:
    """Row of metric summary cards. metrics = [(label, value, color, bg), ...]"""
    cells = []
    for label, value, fg, bg in metrics:
        cell = Table(
            [[Paragraph(label, _style("ml", size=8, color=C_MUTED))],
             [Paragraph(value, _style("mv", size=16, font=_CJK_BOLD, color=fg))]],
            colWidths=[W / len(metrics) - 4],
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("ROUNDEDCORNERS",(0,0),(-1,-1), [4, 4, 4, 4]),
        ]))
        cells.append(cell)

    row = Table([cells], colWidths=[W / len(metrics)] * len(metrics))
    row.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1), 2),
        ("RIGHTPADDING",  (0,0),(-1,-1), 2),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    return row


def _section_divider(title: str) -> Table:
    t = Table(
        [[Paragraph(title, _style("sh", size=10, font=_CJK_BOLD, color=C_MUTED))]],
        colWidths=[W],
    )
    t.setStyle(TableStyle([
        ("LINEBELOW",     (0,0),(-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    return t


def _insight_box(text: str) -> Table:
    t = Table(
        [[Paragraph(text, S_INSIGHT)]],
        colWidths=[W],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_HEADER_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("LINEAFTER",     (0,0),(0,-1),  2, C_MUTED),
    ]))
    return t


def _msg_example(date_sender: str, text: str, sentiment: str = "neu") -> Table:
    accent = {"neg": C_RED, "pos": C_GREEN, "neu": C_BORDER}.get(sentiment, C_BORDER)
    t = Table(
        [[Paragraph(date_sender, S_SMALL)],
         [Paragraph(text, S_BODY)]],
        colWidths=[W - 8],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_HEADER_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("LINEAFTER",     (0,0),(0,-1),  2.5, accent),
    ]))
    return t


def _issue_row(title: str, meta: str, status: str) -> Table:
    is_open = status in ("unresolved", "unclear")
    tag_text = "待處理" if is_open else "已解決"
    tag_style = S_BADGE_R if is_open else S_BADGE_G
    tag_bg    = C_RED_BG  if is_open else C_GREEN_BG
    row_bg    = C_AMBER_BG if is_open else C_WHITE

    tag_cell = Table([[Paragraph(tag_text, tag_style)]], colWidths=[18*mm])
    tag_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), tag_bg),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), [3,3,3,3]),
    ]))
    body = Table(
        [[Paragraph(title, _style("it", size=9, font=_CJK_BOLD))],
         [Paragraph(meta,  S_SMALL)]],
        colWidths=[W - 22*mm - 8],
    )
    body.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 1),
        ("BOTTOMPADDING", (0,0),(-1,-1), 1),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    row = Table([[tag_cell, body]], colWidths=[22*mm, W - 22*mm])
    row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), row_bg),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LINEBELOW",     (0,0),(-1,-1), 0.5, C_BORDER),
    ]))
    return row


def _tripwire_box(reasons: list[str]) -> Table:
    lines = [Paragraph("紅旗警報（Tripwire）", _style("tw", size=9, font=_CJK_BOLD, color=C_RED))]
    for r in reasons:
        lines.append(Paragraph(f"  ● {r}", _style("tr", size=8, color=C_RED, space_after=2)))
    t = Table([[l] for l in lines], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_RED_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("LINEAFTER",     (0,0),(0,-1),  3, C_RED),
    ]))
    return t


def _entropy_chart(
    series: list,
    escalation_fracs: list[float],
    width: float = None,
    height: float = 55,
) -> Drawing:
    """
    Render semantic entropy time series as a sparkline Drawing (platypus flowable).
    X axis: message index (0 → N-1).
    Y axis: entropy in bits.
    Red dashed verticals mark escalation trigger positions.
    """
    width = width or W
    if not series or len(series) < 2:
        return None

    entropies = [e for _, e in series]
    e_min = min(entropies)
    e_max = max(entropies)
    e_range = max(e_max - e_min, 0.5)   # prevent collapse to flat line
    n = len(series)

    PAD_L, PAD_R, PAD_T, PAD_B = 6, 6, 8, 18
    plot_w = width - PAD_L - PAD_R
    plot_h = height - PAD_T - PAD_B

    d = Drawing(width, height)

    # Background
    d.add(Rect(0, 0, width, height, fillColor=C_HEADER_BG, strokeColor=None))
    d.add(Rect(PAD_L, PAD_B, plot_w, plot_h, fillColor=colors.white, strokeColor=C_BORDER, strokeWidth=0.5))

    def _x(i):
        return PAD_L + (i / (n - 1)) * plot_w

    def _y(e):
        return PAD_B + ((e - e_min) / e_range) * plot_h

    # Mean reference line (grey dashed)
    mean_e = sum(entropies) / n
    y_mean = _y(mean_e)
    d.add(Line(PAD_L, y_mean, PAD_L + plot_w, y_mean,
               strokeColor=C_MUTED, strokeWidth=0.5, strokeDashArray=[3, 3]))

    # Escalation verticals (red dashed)
    for frac in escalation_fracs:
        x_esc = PAD_L + frac * plot_w
        d.add(Line(x_esc, PAD_B, x_esc, PAD_B + plot_h,
                   strokeColor=C_RED, strokeWidth=1.0, strokeDashArray=[2, 2]))

    # Entropy polyline
    pts = []
    for i, (_, e) in enumerate(series):
        pts.extend([_x(i), _y(e)])
    d.add(PolyLine(pts, strokeColor=C_BLUE, strokeWidth=1.5, strokeLineJoin=1))

    # X-axis labels: start / mid / end
    label_style = dict(fontName=_CJK_FONT, fontSize=6, fillColor=C_MUTED)
    d.add(GString(PAD_L, PAD_B - 10, "對話開始", **label_style))
    d.add(GString(PAD_L + plot_w - 28, PAD_B - 10, "對話結尾", **label_style))

    # Y-axis min/max annotations
    d.add(GString(PAD_L + plot_w + 2, PAD_B - 2,
                  f"{e_min:.1f}", fontName=_CJK_FONT, fontSize=6, fillColor=C_MUTED))
    d.add(GString(PAD_L + plot_w + 2, PAD_B + plot_h - 4,
                  f"{e_max:.1f} bits", fontName=_CJK_FONT, fontSize=6, fillColor=C_MUTED))

    return d


def _page_header(canvas, doc):
    canvas.saveState()
    canvas.setFont(_CJK_BOLD, 8)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(14*mm, A4[1] - 10*mm, "LINE 群組健康度 Triage 報告")
    canvas.drawRightString(A4[0] - 14*mm, A4[1] - 10*mm,
                           f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    canvas.setStrokeColor(C_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(14*mm, A4[1] - 12*mm, A4[0] - 14*mm, A4[1] - 12*mm)

    canvas.drawCentredString(A4[0]/2, 8*mm, f"第 {doc.page} 頁")
    canvas.restoreState()


def _overview_page(metrics_list: list[GroupMetrics], now: datetime) -> list:
    """第一頁：10 群組一覽表 + 快速行動建議"""
    story = []

    # 大標題
    story.append(Spacer(1, 4))
    title_data = [
        [Paragraph("LINE 群組健康度 Triage 總覽",
                   _style("ov_title", size=20, font=_CJK_BOLD, space_after=2)),
         Paragraph(f"基準時間\n{now.strftime('%Y-%m-%d %H:%M')}",
                   _style("ov_ts", size=9, color=C_MUTED, align=TA_RIGHT, space_after=0))],
    ]
    title_t = Table(title_data, colWidths=[W*0.65, W*0.35])
    title_t.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "BOTTOM"),
        ("LINEBELOW",     (0,0),(-1,0), 0.5, C_BORDER),
        ("BOTTOMPADDING", (0,0),(-1,0), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(title_t)
    story.append(Spacer(1, 10))

    # 統計卡片
    total = len(metrics_list)
    n_red  = sum(1 for m in metrics_list if m.tripwire)
    n_warn = sum(1 for m in metrics_list if not m.tripwire and m.composite >= 0.15)
    n_ok   = total - n_red - n_warn
    story.append(_metric_cards([
        ("群組總數",   str(total),  C_BLUE,  C_BLUE_BG),
        ("紅旗警報",   str(n_red),  C_RED,   C_RED_BG   if n_red  else C_HEADER_BG),
        ("需要關注",   str(n_warn), C_AMBER, C_AMBER_BG if n_warn else C_HEADER_BG),
        ("狀態正常",   str(n_ok),   C_GREEN, C_GREEN_BG),
    ]))
    story.append(Spacer(1, 12))

    # 群組排名一覽表
    story.append(_section_divider("群組排名一覽（依優先處理順序）"))
    story.append(Spacer(1, 4))

    # 表頭
    COL = [8*mm, W*0.30, W*0.13, W*0.13, W*0.13, W*0.13, W*0.10]
    hdr_style = _style("th", size=8, font=_CJK_BOLD, color=C_MUTED, align=TA_CENTER, space_after=0)
    hdr = Table([[
        Paragraph("#",        hdr_style),
        Paragraph("群組",     _style("th2", size=8, font=_CJK_BOLD, color=C_MUTED, space_after=0)),
        Paragraph("綜合分數", hdr_style),
        Paragraph("I1 未回應", hdr_style),
        Paragraph("I2 延遲",  hdr_style),
        Paragraph("I4 負面",  hdr_style),
        Paragraph("狀態",     hdr_style),
    ]], colWidths=COL)
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_HEADER_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("LINEBELOW",     (0,0),(-1,-1), 0.5, C_BORDER),
    ]))
    story.append(hdr)

    def _sev_cell(val: float) -> Paragraph:
        color = _severity_color(val)
        return Paragraph(f"{val:.2f}", _style(f"sc{val}", size=8, color=color, align=TA_CENTER, space_after=0))

    for rank, m in enumerate(metrics_list, 1):
        if m.tripwire:
            status_p = Paragraph("🚨 紅旗", _style("st_r", size=8, font=_CJK_BOLD, color=C_RED, align=TA_CENTER, space_after=0))
            row_bg = C_RED_BG
        elif m.composite >= 0.15:
            status_p = Paragraph("⚠ 關注", _style("st_a", size=8, font=_CJK_BOLD, color=C_AMBER, align=TA_CENTER, space_after=0))
            row_bg = C_AMBER_BG
        else:
            status_p = Paragraph("正常", _style("st_g", size=8, color=C_GREEN, align=TA_CENTER, space_after=0))
            row_bg = C_WHITE

        row = Table([[
            Paragraph(str(rank), _style(f"rk{rank}", size=8, color=C_MUTED, align=TA_CENTER, space_after=0)),
            Paragraph(m.group_id, _style(f"gn{rank}", size=8, font=_CJK_BOLD, space_after=0)),
            _sev_cell(m.composite),
            _sev_cell(m.i1_severity),
            _sev_cell(m.i2_severity),
            _sev_cell(m.i4_severity),
            status_p,
        ]], colWidths=COL)
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), row_bg),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
            ("LINEBELOW",     (0,0),(-1,-1), 0.5, C_BORDER),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(row)

    story.append(Spacer(1, 14))

    # 快速行動建議
    story.append(_section_divider("立即行動建議"))
    story.append(Spacer(1, 4))

    red_groups  = [m for m in metrics_list if m.tripwire]
    warn_groups = [m for m in metrics_list if not m.tripwire and m.composite >= 0.15]

    if red_groups:
        names = "、".join(m.group_id for m in red_groups)
        story.append(_insight_box(
            f"【立即處理】{names}｜已觸發升級詞彙（退款／投訴／找主管），"
            "請在 1 小時內主管介入確認，評估是否需要電話溝通。"
        ))
        story.append(Spacer(1, 6))

    if warn_groups:
        for m in warn_groups:
            flags = []
            if m.i2_severity >= 0.5:
                flags.append(f"回應延遲 P90={m.i2_p90_min:.0f} 分")
            if m.i1_severity >= 0.3:
                flags.append(f"未回覆提問 {m.i1_open_questions} 個（最老 {m.i1_oldest_age_min:.0f} 分）")
            if m.i4_severity >= 0.3:
                flags.append(f"負面情緒比 {m.i4_neg_ratio:.0%}")
            desc = "、".join(flags) if flags else f"綜合分數 {m.composite:.2f}"
            story.append(_insight_box(
                f"【今日跟進】{m.group_id}｜{desc}，建議今日內主動聯繫客戶確認狀態。"
            ))
            story.append(Spacer(1, 4))

    ok_names = "、".join(m.group_id for m in metrics_list if not m.tripwire and m.composite < 0.15)
    if ok_names:
        story.append(_insight_box(f"【維持現狀】{ok_names}｜指標正常，按原節奏推進即可。"))

    return story


def generate_report(
    metrics_list: list[GroupMetrics],
    output_path: str,
    now: datetime | None = None,
    sentiment_examples: dict | None = None,
    latency_examples: dict | None = None,
):
    now = now or datetime.now()

    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=18*mm,  bottomMargin=16*mm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main", topPadding=6, bottomPadding=4,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=_page_header)])

    story = []
    story += _overview_page(metrics_list, now)
    story.append(PageBreak())

    for idx, m in enumerate(metrics_list):
        if idx > 0:
            story.append(Paragraph('<para><br/></para>', S_BODY))

        # --- Report header ---
        flag = "🚨 紅旗警報" if m.tripwire else ""
        header_data = [
            [Paragraph(m.group_id, S_TITLE),
             Paragraph(f"綜合分數\n{m.composite:.2f}", _style("score", size=20, font=_CJK_BOLD,
                       color=_severity_color(m.composite), align=TA_RIGHT, space_after=0))],
            [Paragraph(
                f"分析時間：{now.strftime('%Y-%m-%d %H:%M')}　{flag}",
                _style("sub2", size=9, color=C_MUTED)), ""],
        ]
        hdr = Table(header_data, colWidths=[W*0.72, W*0.28])
        hdr.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("SPAN",          (1,0),(1,1)),
            ("LINEBELOW",     (0,1),(-1,1), 0.5, C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 2),
            ("BOTTOMPADDING", (0,1),(-1,1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))
        story.append(hdr)
        story.append(Spacer(1, 6))

        # --- Tripwire box ---
        if m.tripwire and m.tripwire_reasons:
            story.append(_tripwire_box(m.tripwire_reasons))
            story.append(Spacer(1, 8))

        # --- Metric cards ---
        story.append(_metric_cards([
            ("I1 未回應提問數",   f"{m.i1_open_questions} 題",
             C_GREEN if m.i1_severity < 0.35 else _severity_color(m.i1_severity),
             C_GREEN_BG if m.i1_severity < 0.35 else C_RED_BG),
            ("I2 回應延遲 P90",  f"{m.i2_p90_min:.0f} 分",
             C_GREEN if m.i2_severity < 0.35 else _severity_color(m.i2_severity),
             C_GREEN_BG if m.i2_severity < 0.35 else C_AMBER_BG),
            ("I4 客戶負面情緒比", f"{m.i4_neg_ratio:.0%}",
             _severity_color(m.i4_severity),
             C_RED_BG if m.i4_severity >= 0.35 else C_GREEN_BG),
            ("I5 近24h 訊息量",  f"{m.i5_msg_count_24h} 則",
             C_BLUE, C_BLUE_BG),
        ]))
        story.append(Spacer(1, 10))

        # --- Severity bars ---
        story.append(KeepTogether([
            _section_divider("各指標風險等級（0 = 無風險 / 1 = 最高風險）"),
            Spacer(1, 4),
            _bar_table("I1 未回應提問", m.i1_severity,
                       f"severity {m.i1_severity:.2f}  |  最老 {m.i1_oldest_age_min:.0f} 分鐘"),
            _bar_table("I2 回應延遲",   m.i2_severity,
                       f"severity {m.i2_severity:.2f}  |  P90 = {m.i2_p90_min:.0f} 分鐘"),
            _bar_table("I4 負面情緒",   m.i4_severity,
                       f"severity {m.i4_severity:.2f}  |  負面比 {m.i4_neg_ratio:.1%}"),
            _bar_table("綜合分數",       m.composite,
                       f"composite {m.composite:.3f}"),
            Spacer(1, 6),
        ]))

        # --- I2 detail ---
        ex_lat = (latency_examples or {}).get(m.group_id, [])
        story.append(KeepTogether([
            _section_divider("I2 回應延遲 — 解讀"),
            Spacer(1, 4),
            Paragraph(
                f"P90 延遲：<b>{m.i2_p90_min:.0f} 分鐘</b>（警示線 120 分 / 臨界線 480 分）",
                S_BODY),
            Spacer(1, 4),
        ] + [_msg_example(d, t, s) for d, t, s in ex_lat] + [
            Spacer(1, 4),
            _insight_box(
                "回應延遲 P90 反映業務時間內團隊對客戶提問的回應速度。"
                "P90 越低代表 90% 的問題能在更短時間內被接收到回應。"
                f"目前 P90 = {m.i2_p90_min:.0f} 分鐘，"
                + ("屬於優良水準，無需特別關注。" if m.i2_severity < 0.35
                   else "已超過警示線，建議提升值班頻率。")
            ),
            Spacer(1, 8),
        ]))

        # --- I4 detail ---
        ex_sent = (sentiment_examples or {}).get(m.group_id, [])
        story.append(KeepTogether([
            _section_divider("I4 客戶負面情緒 — 解讀"),
            Spacer(1, 4),
            Paragraph(
                f"近 24 小時客戶訊息中，<b>{m.i4_neg_ratio:.1%}</b> 帶有負面詞彙"
                f"（severity = {m.i4_severity:.2f}）。",
                S_BODY),
            Spacer(1, 4),
        ] + [_msg_example(d, t, s) for d, t, s in ex_sent] + [
            Spacer(1, 4),
            _insight_box(
                "負面情緒指標擷取客戶近 24 小時訊息中的負面語氣詞彙比例。"
                + ("目前比例偏高，建議主動關懷客戶，確認核心訴求是否已被理解。"
                   if m.i4_severity >= 0.35
                   else "目前情緒正常，持續維持良好溝通即可。")
            ),
            Spacer(1, 8),
        ]))

        # --- I6 Semantic Entropy ---
        if len(m.i6_entropy_series) >= 5:
            chart = _entropy_chart(m.i6_entropy_series, m.i6_escalation_fracs)
            slope = m.i6_entropy_slope
            peak_pct = int(m.i6_peak_entropy_frac * 100)
            if slope > 0.5:
                trend_txt = f"語義熵在對話後段明顯上升（slope={slope:+.2f}），對話主題持續擴散或情緒複雜化。"
                trend_color = C_RED
            elif slope > 0.1:
                trend_txt = f"語義熵略有上升（slope={slope:+.2f}），對話內容輕微多元化，可持續觀察。"
                trend_color = C_AMBER
            elif slope >= -0.1:
                trend_txt = f"語義熵維持穩定（slope={slope:+.2f}），對話聚焦、話題一致。"
                trend_color = C_GREEN
            else:
                trend_txt = f"語義熵在後段下降（slope={slope:+.2f}），對話收斂至特定主題。"
                trend_color = C_BLUE

            esc_note = ""
            if m.i6_escalation_fracs:
                esc_positions = "、".join(f"{int(f*100)}%" for f in m.i6_escalation_fracs)
                esc_note = f"  升級詞出現於對話 {esc_positions} 處（圖中紅色虛線）。"

            correlation_note = ""
            if m.i6_escalation_fracs and m.i6_peak_entropy_frac > 0:
                avg_esc = sum(m.i6_escalation_fracs) / len(m.i6_escalation_fracs)
                diff = abs(m.i6_peak_entropy_frac - avg_esc)
                if diff < 0.2:
                    correlation_note = "  熵峰值與升級事件高度吻合，顯示升級前後對話出現明顯語義擴散。"

            story.append(KeepTogether([
                _section_divider("I6 語義熵時間序列（Semantic Entropy）"),
                Spacer(1, 4),
                Paragraph(
                    f"字元 bigram Shannon 熵（bits）隨對話進展的變化。"
                    f"  均值 <b>{m.i6_mean_entropy:.2f} bits</b>，"
                    f"熵峰值出現於對話 <b>{peak_pct}%</b> 處。",
                    S_BODY,
                ),
                Spacer(1, 6),
            ] + ([chart, Spacer(1, 6)] if chart else []) + [
                _insight_box(
                    trend_txt + esc_note + correlation_note
                ),
                Spacer(1, 8),
            ]))

        # --- I3 issues ---
        if m.i3_issues:
            open_issues  = [i for i in m.i3_issues if i.get("status") in ("unresolved", "unclear")]
            closed_issues = [i for i in m.i3_issues if i.get("status") == "resolved"]
            issue_rows = []
            for iss in open_issues + closed_issues:
                typ_map = {"question": "問題", "request": "請求",
                           "complaint": "投訴", "report": "回報"}
                typ = typ_map.get(iss.get("type", ""), iss.get("type", ""))
                raised = iss.get("raised_at", "")[:10]
                meta = f"提出：{raised}　類型：{typ}"
                if iss.get("resolution_evidence"):
                    meta += f"　{iss['resolution_evidence'][:40]}"
                issue_rows.append(_issue_row(
                    iss.get("summary", "（無摘要）"),
                    meta,
                    iss.get("status", "unclear"),
                ))

            story.append(KeepTogether([
                _section_divider("I3 議題處理進度（LLM 抽取）"),
                Spacer(1, 4),
                Paragraph(
                    f"共 {len(m.i3_issues)} 個議題：<b>{len(open_issues)} 個待處理</b>、"
                    f"{len(closed_issues)} 個已解決。"
                    f"最老未解議題年齡：{m.i3_oldest_age_min:.0f} 分鐘。",
                    S_BODY),
                Spacer(1, 6),
            ] + issue_rows + [
                Spacer(1, 4),
                _insight_box(
                    f"共 {len(open_issues)} 項議題尚未定案。"
                    "建議在下次客戶接觸前整理待確認清單，逐項確認後關閉。"
                ),
                Spacer(1, 8),
            ]))

    doc.build(story)
