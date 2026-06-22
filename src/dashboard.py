"""
Render ranked group list to terminal or return as dict for downstream use.
"""
from datetime import datetime
from .metrics import GroupMetrics


def rank_groups(metrics_list: list[GroupMetrics]) -> list[GroupMetrics]:
    """Sort groups: tripwire first (by composite desc), then rest by composite desc."""
    tripwire = sorted([m for m in metrics_list if m.tripwire], key=lambda x: -x.composite)
    normal = sorted([m for m in metrics_list if not m.tripwire], key=lambda x: -x.composite)
    return tripwire + normal


def render_text(ranked: list[GroupMetrics]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"  LINE 群組健康度 Triage 報表  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    for i, m in enumerate(ranked, 1):
        flag = "🚨 [TRIPWIRE]" if m.tripwire else ""
        lines.append(f"\n#{i}  {m.group_id}  {flag}")
        lines.append(f"    綜合分數: {m.composite:.3f}")
        lines.append(f"    I1 未回應提問: severity={m.i1_severity:.2f}  開放題數={m.i1_open_questions}  最老={m.i1_oldest_age_min:.0f}分")
        lines.append(f"    I2 回應延遲P90: severity={m.i2_severity:.2f}  P90={m.i2_p90_min:.0f}分")
        if m.i3_issues:
            lines.append(f"    I3 最老未解議題: severity={m.i3_severity:.2f}  開放={m.i3_open_issues}個  最老={m.i3_oldest_age_min:.0f}分")
            for iss in m.i3_issues:
                status_icon = "✅" if iss.get("status") == "resolved" else "🔴"
                lines.append(f"       {status_icon} [{iss.get('type','')}] {iss.get('summary','')[:50]}")
        if m.i4_neg_ratio >= 0.5:
            mood = "⚠️ 情緒極度異常"
        elif m.i4_neg_ratio >= 0.3:
            mood = "⚠️ 情緒明顯負面"
        elif m.i4_neg_ratio >= 0.1:
            mood = "⚠️ 情緒略顯不穩"
        else:
            mood = "正常"
        lines.append(f"    I4 客戶負面情緒: severity={m.i4_severity:.2f}  負面比={m.i4_neg_ratio:.1%}  [{mood}]")
        lines.append(f"    I5 近24h訊息量: {m.i5_msg_count_24h}")
        if m.i6_entropy_series:
            slope_label = (
                "↑ 上升（對話趨複雜）" if m.i6_entropy_slope > 0.5
                else "↗ 緩升" if m.i6_entropy_slope > 0.1
                else "→ 穩定" if m.i6_entropy_slope >= -0.1
                else "↘ 收斂（對話趨專一）"
            )
            peak_pct = int(m.i6_peak_entropy_frac * 100)
            lines.append(
                f"    I6 語義熵: mean={m.i6_mean_entropy:.2f}bits  "
                f"slope={m.i6_entropy_slope:+.2f}  "
                f"峰值@對話{peak_pct}%  [{slope_label}]"
            )
        if m.tripwire_reasons:
            for r in m.tripwire_reasons:
                lines.append(f"    ⚠️  {r}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
