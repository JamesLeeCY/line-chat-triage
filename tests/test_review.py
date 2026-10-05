import json
from collections import Counter

import numpy as np
import pytest

from src.community.compare import agreement
from src.community.evaluate import evaluate
from src.community.review import build_queue, stratum_weights, summarize


def _samples():
    rows = [{"id": f"t{i}", "split": "test"} for i in range(100)]
    rows += [{"id": f"e{i}", "split": "enrich_test"} for i in range(50)]
    rows += [{"id": f"r{i}", "split": "train"} for i in range(30)]  # never queued
    return rows


def _label(stance):
    return {"stance": stance, "topic": "chit_chat", "tickers": []}


def _sources():
    v1 = {f"t{i}": _label("neutral") for i in range(100)}
    v2 = {f"t{i}": _label("neutral") for i in range(100)}
    v2.update({f"e{i}": _label("neutral") for i in range(50)})
    for i in range(5):            # both directional, agreeing
        v1[f"t{i}"] = v2[f"t{i}"] = _label("bullish")
    for i in range(5, 8):         # disagreement: v1 bearish, v2 neutral
        v1[f"t{i}"] = _label("bearish")
    for i in range(10):           # enrich_test only has v2
        v2[f"e{i}"] = _label("bearish")
    return {"v1": v1, "v2": v2}


def test_queue_takes_all_directional_and_a_proportional_control():
    queue = build_queue(_samples(), _sources(), control=26, seed=1)
    s = summarize(queue)

    assert s["test"]["ai_directional"] == {"queued": 8, "population": 8}
    assert s["test"]["disputed"] == 3
    assert s["enrich_test"]["ai_directional"] == {"queued": 10, "population": 10}  # judged by v2 alone
    # 92 + 40 neutral → control 26 split ~ 18 / 8
    assert s["test"]["ai_neutral"] == {"queued": 18, "population": 92}
    assert s["enrich_test"]["ai_neutral"] == {"queued": 8, "population": 40}
    assert not any(q["split"] == "train" for q in queue)
    assert len({q["id"] for q in queue}) == len(queue)


def test_queue_is_shuffled_and_reproducible():
    a = build_queue(_samples(), _sources(), control=26, seed=1)
    b = build_queue(_samples(), _sources(), control=26, seed=1)
    assert a == b
    first_half = [q["stratum"] for q in a[: len(a) // 2]]
    assert "ai_neutral" in first_half and "ai_directional" in first_half


def test_weights_reconstruct_population_even_when_partly_labelled():
    queue = build_queue(_samples(), _sources(), control=26, seed=1)
    labelled = {q["id"] for q in queue[::2]}  # human has done half the queue
    w = stratum_weights(queue, labelled)
    for split, stratum in {(q["split"], q["stratum"]) for q in queue if q["id"] in labelled}:
        pop = next(q["stratum_population"] for q in queue if (q["split"], q["stratum"]) == (split, stratum))
        total = sum(w[q["id"]] for q in queue if q["id"] in w and (q["split"], q["stratum"]) == (split, stratum))
        assert total == pytest.approx(pop)


def test_weighted_agreement_estimates_full_split():
    # Full test split: 92 AI-neutral (truly neutral except 4 the AI missed), 8 AI-directional (6 correct).
    ids = [f"t{i}" for i in range(100)]
    ai = {i: {"stance": "neutral"} for i in ids}
    truth = {i: {"stance": "neutral"} for i in ids}
    for i in range(8):
        ai[f"t{i}"] = {"stance": "bullish"}
        truth[f"t{i}"] = {"stance": "bullish" if i < 6 else "neutral"}
    for i in range(8, 12):
        truth[f"t{i}"] = {"stance": "bearish"}  # missed by the AI
    full = agreement(truth, ai, "stance")

    # Human labels only the queue: all 8 directional + every other neutral (46 of 92)
    queue = [{"id": f"t{i}", "split": "test", "stratum": "ai_directional", "stratum_population": 8} for i in range(8)]
    queue += [{"id": f"t{i}", "split": "test", "stratum": "ai_neutral", "stratum_population": 92}
              for i in range(8, 100, 2)]
    human = {q["id"]: truth[q["id"]] for q in queue}
    est = agreement(human, ai, "stance", weights=stratum_weights(queue, set(human)))
    raw = agreement(human, ai, "stance")

    assert est["weighted"] and est["n"] == 54
    assert est["accuracy"] == pytest.approx(full["accuracy"])  # 0.94, unbiased
    assert raw["accuracy"] < est["accuracy"]                   # unweighted queue overstates errors


def test_weighted_evaluate_matches_duplicated_rows():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = ["a", "b", "b"]
    weighted = evaluate(["a", "b"], proba, y, sample_weight=[1, 1, 3])
    duplicated = evaluate(["a", "b"], np.vstack([proba[:2]] + [proba[2:]] * 3), ["a", "b", "b", "b", "b"])
    for k in ("accuracy", "macro_f1", "ece", "brier"):
        assert weighted[k] == pytest.approx(duplicated[k])
    assert weighted["confusion"]["matrix"] == duplicated["confusion"]["matrix"]


def test_annotate_follows_queue_order(tmp_path):
    from fastapi.testclient import TestClient

    from src.community.annotate import create_app

    sample = tmp_path / "s.jsonl"
    sample.write_text("\n".join(json.dumps({"id": i, "split": "test", "group": "g", "ts": 1767600000,
                                            "text": i, "previous": [], "reply_to_text": None})
                                for i in ("a", "b", "c", "d")), encoding="utf-8")
    app = create_app(str(sample), str(tmp_path / "human.jsonl"), ids=["c", "a", "zz"])
    items = TestClient(app).get("/api/items").json()["items"]
    assert [it["id"] for it in items] == ["c", "a"]


def test_review_queue_cli_refuses_to_overwrite(tmp_path, monkeypatch):
    from src.community import __main__ as cli

    sample, labels_dir = tmp_path / "s.jsonl", tmp_path / "labels"
    labels_dir.mkdir()
    sample.write_text("\n".join(json.dumps(r) for r in _samples()), encoding="utf-8")
    for name, src in _sources().items():
        (labels_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"id": k, **v}) for k, v in src.items()), encoding="utf-8")
    monkeypatch.setattr(cli, "LABEL_DIR", labels_dir)
    out = tmp_path / "queue.jsonl"

    cli.main(["review-queue", "--sample", str(sample), "--out", str(out), "--control", "26"])
    first = out.read_text(encoding="utf-8")
    assert Counter(json.loads(l)["stratum"] for l in first.splitlines())["ai_directional"] == 18
    with pytest.raises(SystemExit):
        cli.main(["review-queue", "--sample", str(sample), "--out", str(out)])
    cli.main(["review-queue", "--sample", str(sample), "--out", str(out), "--control", "26", "--force"])
    assert out.read_text(encoding="utf-8") == first  # same seed → same queue


def test_keep_adds_earlier_human_labels_to_their_stratum():
    keep = frozenset({"t50", "t60", "t3"})  # two AI-neutral, one already AI-directional
    base = build_queue(_samples(), _sources(), control=26, seed=1)
    queue = build_queue(_samples(), _sources(), control=26, seed=1, keep=keep)
    ids = {q["id"] for q in queue}
    assert keep <= ids
    s = summarize(queue)
    assert s["test"]["ai_neutral"]["queued"] == 18 + len({"t50", "t60"} - {q["id"] for q in base})
    assert s["test"]["ai_directional"]["queued"] == 8  # t3 was already in the census
