"""
Community mode CLI.

    python -X utf8 -m src.community prepare             # export JSON → SQLite (once)
    python -X utf8 -m src.community groups              # what's in the store
    python -X utf8 -m src.community sample              # stratified gold sample
    python -X utf8 -m src.community label --limit 20    # local Qwen3 labels (resumable)
    python -X utf8 -m src.community annotate            # human labels in the browser
    python -X utf8 -m src.community compare             # each labeller vs human
    python -X utf8 -m src.community train --labels qwen3-8b --test-labels human

Label sources live in data/community/labels/<name>.jsonl; `human` is the
annotation page's output, model sources are named after the model.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

DATA = Path("data/community")
DB = str(DATA / "community.db")
SAMPLE = str(DATA / "gold_sample.jsonl")
LABEL_DIR = DATA / "labels"


def _labels_path(name_or_path: str) -> str:
    """A label source by name (labels/<name>.jsonl) or by explicit path."""
    p = Path(name_or_path)
    return str(p) if p.suffix == ".jsonl" or p.exists() else str(LABEL_DIR / f"{name_or_path}.jsonl")


def _available_sources() -> list[str]:
    return sorted(p.stem for p in LABEL_DIR.glob("*.jsonl"))


def _fmt_ts(ts) -> str:
    from ..parser import CONVERSATION_TZ
    return datetime.fromtimestamp(ts, tz=CONVERSATION_TZ).strftime("%Y-%m-%d") if ts else "-"


def cmd_prepare(args):
    from .store import prepare
    DATA.mkdir(parents=True, exist_ok=True)
    for name, n in prepare(args.export, args.db):
        print(f"[prepare] {name}: {n:,} 則")


def cmd_groups(args):
    from .store import CommunityStore
    for gid, name, n, first, last in CommunityStore(args.db).groups():
        print(f"{gid:>12}  {n:>10,} 則  {_fmt_ts(first)} ~ {_fmt_ts(last)}  {name}")


def cmd_sample(args):
    from .gold import sample_gold
    from .store import CommunityStore
    store = CommunityStore(args.db)
    groups = store.groups()
    if args.per_group:
        per_group = {int(k): int(v) for k, v in (p.split("=") for p in args.per_group.split(","))}
    else:  # equal share per group, so a small group isn't drowned by a big one
        per_group = {gid: args.n // len(groups) for gid, *_ in groups}
    records = sample_gold(store, per_group, test_size=args.test, seed=args.seed)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    n_test = sum(r["split"] == "test" for r in records)
    print(f"[sample] {len(records)} 則 → {args.out}（train {len(records) - n_test} / test {n_test}）")
    for gid, n in per_group.items():
        print(f"         group {gid}: {sum(r['group_id'] == gid for r in records)} / {n}")


def cmd_label(args):
    from .gold import label_gold, read_jsonl
    from .labelers import make_labeler

    labeler = make_labeler(args.backend, args.model, effort=args.effort, ollama_url=args.ollama_url)
    local = args.backend == "ollama"
    batch_size = args.batch_size or (10 if local else 25)
    workers = args.workers or (1 if local else 4)  # a CPU-bound Ollama gains nothing from parallel calls
    out = _labels_path(args.labels or labeler.name)
    ids = None if args.split == "all" else {r["id"] for r in read_jsonl(args.sample) if r["split"] == args.split}
    print(f"[label] {args.backend}:{labeler.model} → {out}（每批 {batch_size} 則，{workers} 個併發）")
    stats = label_gold(args.sample, out, labeler=labeler, batch_size=batch_size, workers=workers,
                       limit=args.limit, ids=ids)
    print(f"[label] {stats}")


def cmd_annotate(args):
    import uvicorn

    from .annotate import create_app
    app = create_app(args.sample, _labels_path("human"), split=args.split, limit=args.limit)
    print(f"[annotate] 打開 http://127.0.0.1:{args.port}  （Ctrl+C 結束，標註會即時存檔）")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def cmd_compare(args):
    from .compare import agreement, format_agreement, usable

    ref_path = _labels_path(args.reference)
    reference = usable(ref_path)
    if not reference:
        sys.exit(f"參考標註 {ref_path} 是空的；先用 annotate 標一些")
    candidates = args.candidates.split(",") if args.candidates else         [s for s in _available_sources() if s != args.reference]
    for name in candidates:
        cand = usable(_labels_path(name))
        for field in ("stance", "topic"):
            r = agreement(reference, cand, field)
            print(format_agreement(f"{name} vs {args.reference}", field, r) if r
                  else f"== {name}: 與 {args.reference} 沒有共同標註的訊息")


def cmd_train(args):
    from collections import Counter

    from .classifier import MajorityClass, TfidfLogReg
    from .evaluate import evaluate, format_report
    from .gold import load_gold

    from .compare import usable

    if not args.labels:
        sys.exit(f"請用 --labels 指定訓練用的標註來源，目前有：{', '.join(_available_sources()) or '（無）'}")
    train = [g for g in load_gold(args.sample, _labels_path(args.labels)) if g["split"] == "train"]
    test_labels = usable(_labels_path(args.test_labels or args.labels))
    test = [{**s, **test_labels[s["id"]]} for s in load_gold(args.sample, _labels_path(args.test_labels or args.labels))
            if s["split"] == "test" and s["id"] in test_labels]
    if not train or not test:
        sys.exit(f"標註資料不足：train {len(train)} / test {len(test)}，請先執行 label / annotate")
    print(f"[train] 訓練：{args.labels}（{len(train)} 則）｜考卷：{args.test_labels or args.labels}（{len(test)} 則）")

    report = {}
    for target in args.targets.split(","):
        print(f"\n######## {target}  分布（train）: {dict(Counter(g[target] for g in train))}")
        X_tr, y_tr = [g["text"] for g in train], [g[target] for g in train]
        X_te, y_te = [g["text"] for g in test], [g[target] for g in test]
        report[target] = {}
        for name, model in [("majority", MajorityClass()), ("tfidf_logreg", TfidfLogReg())]:
            model.fit(X_tr, y_tr)
            result = evaluate(model.classes_, model.predict_proba(X_te), y_te)
            report[target][name] = result
            print(format_report(f"{target} / {name}", result))
            if name == "tfidf_logreg" and args.show_features:
                for cls, feats in model.top_features(12).items():
                    print(f"   top[{cls}]: {' · '.join(feats)}")
            if name == "tfidf_logreg" and args.save_dir:
                import joblib
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, Path(args.save_dir) / f"{target}_tfidf_logreg.joblib")

    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[train] 報告已存到 {args.report}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.community", description="社群模式：多空與話題分析")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="把 Telegram 匯出轉進 SQLite")
    p.add_argument("--export", default="data/telegram")
    p.add_argument("--db", default=DB)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("groups", help="列出群組與訊息數")
    p.add_argument("--db", default=DB)
    p.set_defaults(func=cmd_groups)

    p = sub.add_parser("sample", help="分層抽樣標準答案")
    p.add_argument("--db", default=DB)
    p.add_argument("--n", type=int, default=3000, help="總抽樣數（預設各群組平分）")
    p.add_argument("--per-group", default=None, help="自訂各群組數量，如 3366841830=2000,123=1000")
    p.add_argument("--test", type=int, default=500, help="保留作考卷的數量")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=SAMPLE)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("label", help="用 LLM 標註抽樣（可中斷續跑）")
    p.add_argument("--backend", default="ollama", choices=["ollama", "claude"],
                   help="ollama = 地端模型（預設 qwen3:8b）；claude = Anthropic API（預設 Haiku 4.5）")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                   help="只適用 Claude Opus / Sonnet；Haiku 與 Ollama 不支援")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--labels", default=None, help="輸出的標註來源名稱（預設用模型名）")
    p.add_argument("--split", default="all", choices=["all", "train", "test"])
    p.add_argument("--batch-size", type=int, default=None, help="預設 ollama 10、claude 25")
    p.add_argument("--workers", type=int, default=None, help="預設 ollama 1、claude 4")
    p.add_argument("--limit", type=int, default=None, help="只標註前 N 則（先小量試跑）")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("annotate", help="開啟人工標註網頁（本機）")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--port", type=int, default=8770)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("compare", help="比較各標註來源與參考標註（預設人工）的一致性")
    p.add_argument("--reference", default="human")
    p.add_argument("--candidates", default=None, help="逗號分隔；預設 labels/ 下所有其他來源")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("train", help="訓練分類器並在考卷上評估")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--labels", default=None, help="訓練用的標註來源，如 qwen3-8b")
    p.add_argument("--test-labels", default=None, help="考卷用的標註來源，如 human（預設同 --labels）")
    p.add_argument("--targets", default="stance,topic")
    p.add_argument("--report", default=str(DATA / "eval_report.json"))
    p.add_argument("--save-dir", default=str(DATA / "models"))
    p.add_argument("--show-features", action="store_true", help="列出各類別最具代表性的 n-gram")
    p.set_defaults(func=cmd_train)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
