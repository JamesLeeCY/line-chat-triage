"""
LINE 群組健康度 Triage 系統

Usage:
    python main.py [--conversations data/conversations] [--employees data/employees.txt]
                   [--now YYYY-MM-DDTHH:MM] [--llm] [--llm-threshold 0.3]
                   [--report] [--report-dir reports]
"""
import argparse
import io
import sys
from datetime import datetime

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

from src.parser import load_employees, parse_file, merge_fragments
from src.enrichment import enrich
from src.metrics import compute_metrics
from src.dashboard import rank_groups, render_text


def process_group(filepath: str, employees: set[str], now: datetime):
    group_id, messages = parse_file(filepath, employees)
    messages = merge_fragments(messages, window_secs=60)
    messages = enrich(messages)
    return compute_metrics(group_id, messages, now=now), messages


def run_llm_extraction(metrics_list, messages_map, threshold: float):
    """Phase 2: Run LLM issue extraction on groups that exceed threshold or tripwire."""
    from src.llm_extractor import extract_issues, compute_i3

    for m in metrics_list:
        should_run = m.tripwire or m.composite >= threshold
        if not should_run:
            continue

        messages = messages_map.get(m.group_id, [])
        if not messages:
            continue

        print(f"  [LLM] 抽取議題: {m.group_id} (composite={m.composite:.3f}) ...", end=" ", flush=True)
        try:
            issues = extract_issues(m.group_id, messages)
            i3_sev, i3_age = compute_i3(issues, m.computed_at)
            m.i3_issues = issues
            m.i3_severity = i3_sev
            m.i3_oldest_age_min = i3_age
            m.i3_open_issues = sum(
                1 for iss in issues if iss.get("status") in ("unresolved", "unclear")
            )
            # Re-compute composite with I3
            m.composite = max(
                0.35 * m.i1_severity + 0.20 * m.i2_severity
                + 0.15 * m.i3_severity + 0.30 * m.i4_severity,
                0.95 if m.tripwire else 0.0,
            )
            print(f"找到 {len(issues)} 個議題")
        except Exception as e:
            print(f"失敗: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="LINE 群組 Triage 系統")
    parser.add_argument("--conversations", default="data/conversations", help="對話檔資料夾")
    parser.add_argument("--employees", default="data/employees.txt", help="員工清單")
    parser.add_argument("--now", default=None, help="模擬當前時間 (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--llm", action="store_true", help="啟用 Phase 2 LLM 議題抽取")
    parser.add_argument("--llm-threshold", type=float, default=0.3,
                        help="composite 分數門檻，超過才呼叫 LLM (預設 0.3)")
    parser.add_argument("--report", action="store_true", help="產生 PDF 報告")
    parser.add_argument("--report-dir", default="reports", help="PDF 輸出資料夾 (預設 reports/)")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    employees = load_employees(args.employees)

    conv_dir = Path(args.conversations)
    files = list(conv_dir.glob("*.txt"))
    if not files:
        print(f"找不到對話檔於 {conv_dir}", file=sys.stderr)
        sys.exit(1)

    all_metrics = []
    messages_map = {}
    for f in files:
        try:
            m, messages = process_group(str(f), employees, now)
            all_metrics.append(m)
            messages_map[m.group_id] = messages
            print(f"[OK] {f.name}: {len(m.tripwire_reasons)} tripwire, composite={m.composite:.3f}")
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}", file=sys.stderr)

    if args.llm:
        print(f"\n[Phase 2] LLM 議題抽取 (門檻={args.llm_threshold}) ...")
        run_llm_extraction(all_metrics, messages_map, threshold=args.llm_threshold)

    ranked = rank_groups(all_metrics)
    print()
    print(render_text(ranked))

    if args.report:
        _generate_pdf(ranked, messages_map, now, args.report_dir)


def _generate_pdf(ranked, messages_map, now, report_dir):
    from src.report_writer import generate_report

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(out_dir / f"triage_report_{ts}.pdf")

    # Build sentiment & latency examples from actual messages
    sentiment_examples = {}
    latency_examples  = {}

    for m in ranked:
        msgs = messages_map.get(m.group_id, [])

        # Pick up to 3 negative customer messages for sentiment section
        neg_msgs = [
            msg for msg in msgs
            if msg.role == "customer" and (msg.sentiment or 0) < -0.1
            and not msg.is_sticker
        ][:3]
        sentiment_examples[m.group_id] = [
            (msg.timestamp.strftime("%m/%d %H:%M  客戶"), msg.text[:80], "neg")
            for msg in neg_msgs
        ]

        # Pick one question + staff reply pair for latency section
        lat_ex = []
        for i, msg in enumerate(msgs):
            if msg.role == "customer" and msg.dialogue_act in ("question", "request") and not msg.is_sticker:
                for later in msgs[i+1:]:
                    if later.role == "staff":
                        lat_ex.append((
                            f"{msg.timestamp.strftime('%m/%d %H:%M')}  客戶",
                            msg.text[:80], "neu",
                        ))
                        lat_ex.append((
                            f"{later.timestamp.strftime('%m/%d %H:%M')}  成員（{int((later.timestamp - msg.timestamp).total_seconds() // 60)} 分鐘後）",
                            later.text[:80], "pos",
                        ))
                        break
                if lat_ex:
                    break
        latency_examples[m.group_id] = lat_ex

    print(f"\n[Report] 產生 PDF 報告中 ...")
    generate_report(
        ranked,
        out_path,
        now=now,
        sentiment_examples=sentiment_examples,
        latency_examples=latency_examples,
    )
    print(f"[Report] 已儲存至 {out_path}")


if __name__ == "__main__":
    main()
