"""
Synthetic manager-triage benchmark: A generates groups with planted risks,
B predicts alerts (rule system and a local LLM), C validates and scores.

    python -X utf8 -m src.synth bench --threads 2                 # measure local speed on one group
    python -X utf8 -m src.synth generate --n 10 --threads 2       # A: qwen3 writes the conversations
    python -X utf8 -m src.synth predict --threads 2               # B: rule system + phi4
    python -X utf8 -m src.synth validate                          # C: Claude checks fidelity (needs credentials)
    python -X utf8 -m src.synth score                             # predictors vs planted truth

Each run lives in data/synth/<run>/ (excluded from git): conversations/,
truth.jsonl, employees.txt, pred_rule.jsonl, pred_llm.jsonl, fidelity.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from .scenarios import build_scenarios

DATA = Path("data/synth")


def _run_dir(args) -> Path:
    return DATA / args.run


def _truth(run: Path) -> list[dict]:
    path = run / "truth.jsonl"
    if not path.exists():
        sys.exit(f"找不到 {path}，請先執行 generate")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ollama(model: str, threads, temperature: float = 0.0):
    from ..community.labelers import OllamaLabeler

    think = False if model.startswith(("qwen3", "deepseek-r1")) else None
    return OllamaLabeler(model, think=think, threads=threads, temperature=temperature)


def cmd_generate(args):
    from .generate import generate

    scenarios = build_scenarios(args.total, seed=args.seed)[: args.n]
    model = _ollama(args.model, args.threads, temperature=0.7)
    print(f"[generate] {args.model}（threads={args.threads or '預設'}）→ {_run_dir(args)}，{len(scenarios)} 個群組")
    print(f"[generate] {generate(scenarios, model, str(_run_dir(args)))}")


def cmd_predict(args):
    from .predict import predict_llm, predict_rule

    run = _run_dir(args)
    truth = _truth(run)
    rule = predict_rule(str(run / "conversations"), truth, str(run / "employees.txt"))
    (run / "pred_rule.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rule),
                                         encoding="utf-8")
    print(f"[predict] 規則系統：{len(rule)} 個群組 → {run / 'pred_rule.jsonl'}")
    if not args.skip_llm:
        print(f"[predict] {args.model}（threads={args.threads or '預設'}）")
        llm = predict_llm(str(run / "conversations"), truth, _ollama(args.model, args.threads),
                          str(run / "pred_llm.jsonl"))
        print(f"[predict] LLM：{len(llm)} 個群組 → {run / 'pred_llm.jsonl'}")


def cmd_validate(args):
    from ..community.labelers import ClaudeLabeler
    from .validate import check_fidelity

    run = _run_dir(args)
    judge = ClaudeLabeler(args.model, effort=args.effort)
    rows = check_fidelity(str(run), _truth(run), judge, str(run / "fidelity.jsonl"))
    print(f"[validate] {sum(r['faithful'] for r in rows)}/{len(rows)} 個群組符合劇本 → {run / 'fidelity.jsonl'}")


def cmd_score(args):
    from .validate import format_scores, score

    run = _run_dir(args)
    truth = _truth(run)
    fid_path = run / "fidelity.jsonl"
    faithful = None
    if fid_path.exists():
        faithful = {r["group"] for r in map(json.loads, fid_path.read_text(encoding="utf-8").splitlines())
                    if r["faithful"]}
    results = {}
    for name in ("rule", "llm"):
        path = run / f"pred_{name}.jsonl"
        if not path.exists():
            continue
        preds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        results[name] = {"all": score(truth, preds)}
        print(format_scores(f"B-{name}，全部群組", results[name]["all"]))
        if faithful is not None:
            results[name]["faithful"] = score([t for t in truth if t["group"] in faithful], preds)
            print(format_scores(f"B-{name}，只計 C 判定符合劇本的群組", results[name]["faithful"]))
    if faithful is None:
        print("[score] 尚未執行 validate：以上分數未排除不符劇本的群組")
    (run / "scores.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float),
                                     encoding="utf-8")


def cmd_bench(args):
    """One real group per step at the given thread count; projects 10 / 100 groups."""
    from .generate import render_line_export, write_text
    from .predict import SYSTEM as PRED_SYSTEM
    from .predict import LLMVerdict

    sc = build_scenarios(1, seed=args.seed)[0]
    results = {}

    gen = _ollama(args.gen_model, args.threads, temperature=0.7)
    t0 = time.time()
    texts, qa = write_text(sc, gen)
    wall = time.time() - t0
    calls = qa["attempt_log"]
    results["A"] = {"per_group_min": round(wall / 60, 1), "calls": len(calls),
                    "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
                    "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
                    "flagged_after_retries": len(qa["role_flips"])}
    print(f"[bench] A {args.gen_model}（threads={args.threads}）：{len(calls)} 次呼叫、"
          f"輸入 {results['A']['prompt_tokens']} / 輸出 {results['A']['output_tokens']} token，"
          f"{wall / 60:.1f} 分（含載入模型），重寫後仍有 {len(qa['role_flips'])} 則錯位或口吻不符")
    conv = render_line_export(sc, texts)
    sample = DATA / "bench" / f"{sc.group}.txt"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(conv, encoding="utf-8")

    pred = _ollama(args.pred_model, args.threads)
    t0 = time.time()
    pred.ask(PRED_SYSTEM, f"現在時間：{sc.now.replace('T', ' ')}\n\n對話：\n{conv}", LLMVerdict)
    wall = time.time() - t0
    st = pred.last_stats
    results["B"] = {"per_group_min": round(wall / 60, 1), "calls": 1,
                    "prompt_tokens": st["prompt_eval_count"], "output_tokens": st["eval_count"]}
    print(f"[bench] B {args.pred_model}（threads={args.threads}）：輸入 {st['prompt_eval_count']} / "
          f"輸出 {st['eval_count']} token，{wall / 60:.1f} 分（含載入模型）")
    for step, r in results.items():
        per = r["per_group_min"]
        print(f"[bench] {step}：每群組約 {per:.1f} 分 → 10 群組約 {per * 10 / 60:.1f} 小時，"
              f"100 群組約 {per * 100 / 60:.1f} 小時")
    out = DATA / f"bench_threads{args.threads}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] 結果已存到 {out}；生成的對話範例：{sample}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.synth", description="合成主管群組：生成 / 預測 / 驗證")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model_default):
        p.add_argument("--run", default="pilot", help="資料夾名稱 data/synth/<run>（預設 pilot）")
        p.add_argument("--model", default=model_default)
        p.add_argument("--threads", type=int, default=None, help="Ollama 最多使用的 CPU 執行緒數（如 2）")

    p = sub.add_parser("generate", help="A：依劇本生成對話與標準答案（可續跑）")
    common(p, "qwen3:8b")
    p.add_argument("--n", type=int, default=10, help="這次生成前 N 個劇本")
    p.add_argument("--total", type=int, default=100, help="劇本總數（固定，前 N 個是試跑子集）")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("predict", help="B：規則系統與本機 LLM 預測（LLM 可續跑）")
    common(p, "phi4")
    p.add_argument("--skip-llm", action="store_true", help="只跑規則系統")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("validate", help="C：Claude 檢查對話是否符合劇本（需 Claude 憑證）")
    p.add_argument("--run", default="pilot")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("score", help="B 對標準答案評分（有 validate 結果時另報只計符合劇本的群組）")
    p.add_argument("--run", default="pilot")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("bench", help="以指定執行緒數實測 A、B 各一個群組的速度")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--gen-model", default="qwen3:8b")
    p.add_argument("--pred-model", default="phi4")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
