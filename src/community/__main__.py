"""
Community mode CLI.

    python -X utf8 -m src.community prepare             # export JSON → SQLite (once)
    python -X utf8 -m src.community groups              # what's in the store
    python -X utf8 -m src.community sample              # stratified gold sample
    python -X utf8 -m src.community label --limit 20    # local Qwen3 labels (resumable)
    python -X utf8 -m src.community critique --labels qwen3-8b-v3 --model phi4 --queue  # AI critic revises labels
    python -X utf8 -m src.community annotate            # human labels in the browser
    python -X utf8 -m src.community compare             # each labeller vs human
    python -X utf8 -m src.community train --labels qwen3-8b --test-labels human
    python -X utf8 -m src.community series --freq 1h --stance-labels qwen3-8b-v2  # entropy time series
    python -X utf8 -m src.community predict --freq 4h        # does entropy predict the group's next window?

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
QUEUE = str(DATA / "review_queue.jsonl")


def _queue_weights(queue_path: str, labelled_ids: set) -> "dict | None":
    """Stratum weights for human-labelled review-queue items, or None without a queue."""
    from .gold import read_jsonl
    from .review import stratum_weights
    queue = read_jsonl(queue_path)
    return stratum_weights(queue, labelled_ids) if queue else None


def _labels_path(name_or_path: str) -> str:
    """A label source by name (labels/<name>.jsonl) or by explicit path."""
    p = Path(name_or_path)
    return str(p) if p.suffix == ".jsonl" or p.exists() else str(LABEL_DIR / f"{name_or_path}.jsonl")


def is_train_split(split: str) -> bool:
    return split == "train" or split.endswith("_train")


def test_splits(samples: list[dict]) -> list[str]:
    """Held-out splits, random test first: never trained on, always scored separately."""
    found = {r["split"] for r in samples if r["split"] == "test" or r["split"].endswith("_test")}
    return sorted(found, key=lambda s: (s != "test", s))


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
    from collections import Counter

    from .gold import _eligible_directional, read_jsonl, sample_gold
    from .store import CommunityStore
    store = CommunityStore(args.db)
    groups = store.groups()
    n = args.n or (1000 if args.enrich else 3000)
    test = args.test if args.test is not None else (200 if args.enrich else 500)
    if args.per_group:
        per_group = {int(k): int(v) for k, v in (p.split("=") for p in args.per_group.split(","))}
    else:  # equal share per group, so a small group isn't drowned by a big one
        per_group = {gid: n // len(groups) for gid, *_ in groups}

    if args.enrich:
        # Append a directional-prior round to the existing sample; earlier rounds
        # and their labels stay untouched. Round 1 is "enrich_*", later rounds
        # "enrich<N>_*" (round 2+ usually skip a test split: enrich_test stays the
        # fixed yardstick so scores remain comparable).
        existing = read_jsonl(args.out)
        if not existing:
            sys.exit(f"{args.out} 不存在，請先執行一般的 sample")
        prefix = "enrich" if args.round == 1 else f"enrich{args.round}"
        if any(r["split"].startswith(prefix + "_") for r in existing):
            sys.exit(f"第 {args.round} 輪加強抽樣已經做過了；要重抽請先手動移除 {prefix}_* 的資料")
        records = sample_gold(store, per_group, test_size=test, seed=args.seed + args.round,
                              eligible=_eligible_directional, exclude=frozenset(r["id"] for r in existing),
                              splits=(f"{prefix}_test", f"{prefix}_train"))
        with open(args.out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    else:
        records = sample_gold(store, per_group, test_size=test, seed=args.seed)
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    print(f"[sample] {len(records)} 則 → {args.out}  {dict(Counter(r['split'] for r in records))}")
    for gid, quota in per_group.items():
        print(f"         group {gid}: {sum(r['group_id'] == gid for r in records)} / {quota}")


def _label_splits(args, labeler, out: str, batch_size: int, workers: int):
    """Run `labeler` over the requested splits (optionally only review-queue ids), resumably."""
    from .gold import label_gold, read_jsonl

    samples = read_jsonl(args.sample)
    if args.queue:
        queued = {q["id"] for q in read_jsonl(QUEUE)}
        if not queued:
            sys.exit("還沒有待標清單，請先執行 review-queue")
        samples = [r for r in samples if r["id"] in queued]
    # Splits run in the order given, so e.g. "test,enrich_test,enrich_train" finishes the
    # small held-out sets first and they can be compared while training labels continue
    splits = sorted({r["split"] for r in samples}) if args.split == "all" else args.split.split(",")
    for split in splits:
        ids = {r["id"] for r in samples if r["split"] == split}
        if not ids:
            print(f"[label] {split}: 抽樣檔裡沒有這個 split，略過", file=sys.stderr)
            continue
        print(f"[label] ---- {split}（{len(ids)} 則）", file=sys.stderr)
        stats = label_gold(args.sample, out, labeler=labeler, batch_size=batch_size, workers=workers,
                           limit=args.limit, ids=ids)
        print(f"[label] {split}: {stats}")


def _batching(args) -> tuple[int, int]:
    local = args.backend == "ollama"
    # a CPU-bound Ollama gains nothing from parallel calls
    return args.batch_size or (10 if local else 25), args.workers or (1 if local else 4)


def cmd_label(args):
    from .labelers import make_labeler

    labeler = make_labeler(args.backend, args.model, effort=args.effort, ollama_url=args.ollama_url,
                           prompt=args.prompt, threads=args.threads)
    batch_size, workers = _batching(args)
    out = _labels_path(args.labels or labeler.name)
    print(f"[label] {args.backend}:{labeler.model} prompt {args.prompt} → {out}"
          f"（每批 {batch_size} 則，{workers} 個併發）")
    _label_splits(args, labeler, out, batch_size, workers)


def cmd_critique(args):
    from .critique import Critic
    from .gold import read_jsonl
    from .labelers import make_labeler

    base = {r["id"]: r for r in read_jsonl(_labels_path(args.labels))}
    if not base:
        sys.exit(f"找不到標註來源 {args.labels}，目前有：{', '.join(_available_sources()) or '（無）'}")
    judge = make_labeler(args.backend, args.model, effort=args.effort, ollama_url=args.ollama_url,
                         threads=args.threads)
    critic = Critic(judge, base, Path(args.labels).stem)
    batch_size, workers = _batching(args)
    out = _labels_path(args.out or critic.name)
    print(f"[critique] 裁判 {args.backend}:{judge.model} 審查 {args.labels} → {out}"
          f"（每批 {batch_size} 則，{workers} 個併發）")
    _label_splits(args, critic, out, batch_size, workers)
    done = [r for r in read_jsonl(out) if "critique" in r]
    print(f"[critique] 累計審查 {len(done)} 則，改判 {sum(r['critique']['changed'] for r in done)} 則")


def cmd_review_queue(args):
    from .gold import read_jsonl
    from .review import build_queue, summarize

    if Path(args.out).exists() and not args.force:
        sys.exit(f"{args.out} 已存在。已標的人工標註依這份清單加權，換清單會讓權重失效；確定要重建請加 --force")
    names = args.sources.split(",") if args.sources else [s for s in _available_sources() if s != "human"]
    sources = {n: {r["id"]: r for r in read_jsonl(_labels_path(n))} for n in names}
    # Earlier human labels came from the random test split, so they can join their
    # stratum without breaking the weights
    keep = frozenset(r["id"] for r in read_jsonl(_labels_path("human")))
    queue = build_queue(read_jsonl(args.sample), sources, control=args.control, seed=args.seed, keep=keep)
    Path(args.out).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queue), encoding="utf-8")
    print(f"[review-queue] 依據 {', '.join(names)} 產生 {len(queue)} 則 → {args.out}")
    for split, info in summarize(queue).items():
        d, n = info.get("ai_directional", {}), info.get("ai_neutral", {})
        print(f"   {split:<12} AI 判多空 {d.get('queued', 0)}（全收，其中意見不一 {info['disputed']}）"
              f"｜AI 判中立 抽 {n.get('queued', 0)} / {n.get('population', 0)} 作對照")


def cmd_annotate(args):
    import uvicorn

    from .annotate import create_app
    from .gold import read_jsonl
    ids = None
    if args.queue:
        queue = read_jsonl(QUEUE)
        if not queue:
            sys.exit("還沒有待標清單，請先執行 review-queue")
        ids = [q["id"] for q in queue]
    app = create_app(args.sample, _labels_path("human"), split=args.split, limit=args.limit, ids=ids)
    print(f"[annotate] 打開 http://127.0.0.1:{args.port}  （Ctrl+C 結束，標註會即時存檔）")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def cmd_compare(args):
    from .compare import agreement, format_agreement, usable
    from .gold import read_jsonl

    ref_path = _labels_path(args.reference)
    reference = usable(ref_path)
    if not reference:
        sys.exit(f"參考標註 {ref_path} 是空的；先用 annotate 標一些")
    if args.candidates:
        candidates = args.candidates.split(",")
    else:
        candidates = [s for s in _available_sources() if s != args.reference]
    by_split: dict[str, set] = {}
    for r in read_jsonl(args.sample):
        by_split.setdefault(r["split"], set()).add(r["id"])

    # Human labels from the review queue are a stratified sample, so they are
    # re-weighted to estimate the full split; other references are used as-is.
    weights = _queue_weights(args.queue, set(reference)) if args.reference == "human" else None

    # Random test = realistic mix; enrich_test = mostly directional messages,
    # where bull/bear errors actually show up. Reported separately, never pooled.
    for split in test_splits(read_jsonl(args.sample)):
        ids = by_split.get(split, set())
        for name in candidates:
            cand = usable(_labels_path(name))
            for field in ("stance", "topic"):
                r = agreement(reference, cand, field, ids=ids, weights=weights)
                if r:
                    print(format_agreement(f"[{split}] {name} vs {args.reference}", field, r))


def cmd_train(args):
    from collections import Counter

    from .classifier import MajorityClass, TfidfLogReg
    from .evaluate import evaluate, format_report
    from .gold import load_gold

    from .compare import usable
    from .gold import read_jsonl

    if not args.labels:
        sys.exit(f"請用 --labels 指定訓練用的標註來源，目前有：{', '.join(_available_sources()) or '（無）'}")
    # Train on the random train split plus every enrichment round, so the classifier
    # sees enough bull / bear examples; test splits are never trained on
    train = [g for g in load_gold(args.sample, _labels_path(args.labels)) if is_train_split(g["split"])]
    test_source = args.test_labels or args.labels
    test_labels = usable(_labels_path(test_source))
    samples = read_jsonl(args.sample)
    tests = {
        split: [{**s, **test_labels[s["id"]]} for s in samples if s["split"] == split and s["id"] in test_labels]
        for split in test_splits(samples)
    }
    tests = {k: v for k, v in tests.items() if v}
    # Human test labels come from the stratified review queue: score with stratum
    # weights so the numbers estimate the full split, not the over-sampled queue
    weights = _queue_weights(args.queue, set(test_labels)) if test_source == "human" else None
    if weights is not None:
        tests = {k: [g for g in v if g["id"] in weights] for k, v in tests.items()}
        tests = {k: v for k, v in tests.items() if v}
    if not train or not tests:
        sys.exit(f"標註資料不足：train {len(train)} / 考卷 {sum(map(len, tests.values()))}，請先執行 label / annotate")
    print(f"[train] 訓練：{args.labels}（{len(train)} 則）｜考卷：{test_source}"
          f"（{'、'.join(f'{k} {len(v)} 則' for k, v in tests.items())}）")

    report = {}
    for target in args.targets.split(","):
        print(f"\n######## {target}  分布（train）: {dict(Counter(g[target] for g in train))}")
        X_tr, y_tr = [g["text"] for g in train], [g[target] for g in train]
        report[target] = {}
        for name, model in [("majority", MajorityClass()), ("tfidf_logreg", TfidfLogReg())]:
            model.fit(X_tr, y_tr)
            report[target][name] = {}
            for split, test in tests.items():
                X_te, y_te = [g["text"] for g in test], [g[target] for g in test]
                w = [weights[g["id"]] for g in test] if weights is not None else None
                result = evaluate(model.classes_, model.predict_proba(X_te), y_te, sample_weight=w)
                report[target][name][split] = result
                print(format_report(f"[{split}] {target} / {name}", result))
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


def cmd_series(args):
    import csv

    from .store import CommunityStore
    from .timeseries import build_series

    model = None
    if args.stance_labels:
        from .classifier import TfidfLogReg
        from .gold import load_gold

        train = [g for g in load_gold(args.sample, _labels_path(args.stance_labels)) if is_train_split(g["split"])]
        if not train:
            sys.exit(f"{args.stance_labels} 沒有訓練用 split 的標註")
        model = TfidfLogReg().fit([g["text"] for g in train], [g["stance"] for g in train])
        print(f"[series] 多空分類器：{args.stance_labels} 的 {len(train)} 則訓練資料（暫定，品質見 HANDOFF）")

    store = CommunityStore(args.db)
    groups = [g for g in store.groups() if args.group is None or g[0] == args.group]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, name, n, _, _ in groups:
        rows = build_series(store, gid, freq=args.freq, stance_model=model)
        if not rows:
            continue
        path = out_dir / f"{gid}_{args.freq}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        active = sum(r["n_msgs"] > 0 for r in rows)
        with_entity = sum(r["n_entity_msgs"] for r in rows) / max(1, sum(r["n_text"] for r in rows))
        print(f"[series] {gid}：{len(rows)} 個 {args.freq} 時間窗（有訊息 {active}），"
              f"提到標的的文字訊息 {with_entity:.0%} → {path}")


def cmd_predict(args):
    from .predict import format_results, load_series, run

    path = Path(args.series_dir) / f"{args.group}_{args.freq}.csv"
    if not path.exists():
        sys.exit(f"找不到 {path}，請先執行 series --freq {args.freq}")
    results = run(load_series(str(path)), args.freq, n_splits=args.splits)
    print(f"[predict] {path}（walk-forward {args.splits} 折，預測下一個 {args.freq} 時間窗）")
    print(format_results(results))
    out = DATA / f"predict_{args.group}_{args.freq}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[predict] 結果已存到 {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.community", description="社群模式：多空與話題分析")
    from .gold import PROMPTS

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
    p.add_argument("--enrich", action="store_true",
                   help="加強抽樣：只挑可能有多空看法的訊息，附加到現有抽樣檔（enrich_train / enrich_test）")
    p.add_argument("--n", type=int, default=None, help="總抽樣數，預設 3000（--enrich 時 1000），各群組平分")
    p.add_argument("--per-group", default=None, help="自訂各群組數量，如 3366841830=2000,123=1000")
    p.add_argument("--test", type=int, default=None, help="保留作考卷的數量，預設 500（--enrich 時 200）")
    p.add_argument("--round", type=int, default=1, help="加強抽樣第幾輪（2 起存成 enrich<N>_train / _test）")
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
    p.add_argument("--threads", type=int, default=None, help="Ollama 最多使用的 CPU 執行緒數（如 2）；預設由 Ollama 決定")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--prompt", default="v2", choices=sorted(PROMPTS), help="提示詞版本（v1 結果存在不帶版本的檔名）")
    p.add_argument("--labels", default=None, help="輸出的標註來源名稱（預設用模型名＋提示詞版本）")
    p.add_argument("--split", default="all",
                   help="all，或依序執行的 split 清單，如 test,enrich_test,enrich_train")
    p.add_argument("--batch-size", type=int, default=None, help="預設 ollama 10、claude 25")
    p.add_argument("--workers", type=int, default=None, help="預設 ollama 1、claude 4")
    p.add_argument("--limit", type=int, default=None, help="只標註前 N 則（先小量試跑）")
    p.add_argument("--queue", action="store_true",
                   help="只標待標清單（review_queue.jsonl）裡的訊息：新提示詞可直接和人工標註比較")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("critique", help="AI 批改迴圈：裁判模型依憑法審查既有標註並改判（可中斷續跑）")
    p.add_argument("--labels", required=True, help="要審查的標註來源，如 qwen3-8b-v3")
    p.add_argument("--backend", default="ollama", choices=["ollama", "claude"])
    p.add_argument("--model", default=None, help="裁判模型，如 phi4；預設 ollama qwen3:8b、claude Haiku 4.5")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--threads", type=int, default=None, help="Ollama 最多使用的 CPU 執行緒數（如 2）；預設由 Ollama 決定")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--out", default=None, help="輸出來源名稱（預設 <labels>+critic-<裁判模型>）")
    p.add_argument("--split", default="all")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--queue", action="store_true", help="只審查待標清單裡的訊息（可直接和人工比較）")
    p.set_defaults(func=cmd_critique)

    p = sub.add_parser("series", help="時間序列：每個時間窗的訊息量、標的 entropy、話題轉移、多空分歧")
    p.add_argument("--db", default=DB)
    p.add_argument("--group", type=int, default=None, help="只算某個群組 id（預設全部）")
    p.add_argument("--freq", default="1h", choices=["1h", "4h", "1d"])
    p.add_argument("--stance-labels", default=None,
                   help="用這個標註來源的訓練 split 訓練多空分類器並加入多空欄位，如 qwen3-8b-v2；省略則不算多空")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--out-dir", default=str(DATA / "series"))
    p.set_defaults(func=cmd_series)

    p = sub.add_parser("predict", help="預測下一時間窗的爆量 / 話題轉移 / 情緒反轉，並比較 entropy 特徵的貢獻")
    p.add_argument("--group", type=int, default=3366841830, help="群組 id（預設台股群）")
    p.add_argument("--freq", default="4h", choices=["1h", "4h", "1d"])
    p.add_argument("--splits", type=int, default=5, help="walk-forward 折數")
    p.add_argument("--series-dir", default=str(DATA / "series"))
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("review-queue", help="產生人工待標清單：AI 判多空的全收＋AI 判中立的抽樣對照")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--sources", default=None, help="依據的 AI 標註來源，逗號分隔；預設 labels/ 下除 human 外全部")
    p.add_argument("--control", type=int, default=60, help="AI 判中立的對照組總數，依各考卷中立數比例分配")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=QUEUE)
    p.add_argument("--force", action="store_true", help="覆蓋既有清單（會讓已標的加權失效）")
    p.set_defaults(func=cmd_review_queue)

    p = sub.add_parser("annotate", help="開啟人工標註網頁（本機）")
    p.add_argument("--queue", action="store_true", help="只標待標清單（review-queue 產生）中的訊息")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--split", default="test", help="要標的 split，如 test、enrich_test")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--port", type=int, default=8770)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("compare", help="比較各標註來源與參考標註（預設人工）的一致性")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--queue", default=QUEUE, help="人工標註的待標清單，用於加權")
    p.add_argument("--reference", default="human")
    p.add_argument("--candidates", default=None, help="逗號分隔；預設 labels/ 下所有其他來源")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("train", help="訓練分類器並在考卷上評估")
    p.add_argument("--sample", default=SAMPLE)
    p.add_argument("--labels", default=None, help="訓練用的標註來源，如 qwen3-8b")
    p.add_argument("--test-labels", default=None, help="考卷用的標註來源，如 human（預設同 --labels）")
    p.add_argument("--queue", default=QUEUE, help="人工標註的待標清單，考卷為 human 時用於加權")
    p.add_argument("--targets", default="stance,topic")
    p.add_argument("--report", default=str(DATA / "eval_report.json"))
    p.add_argument("--save-dir", default=str(DATA / "models"))
    p.add_argument("--show-features", action="store_true", help="列出各類別最具代表性的 n-gram")
    p.set_defaults(func=cmd_train)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
