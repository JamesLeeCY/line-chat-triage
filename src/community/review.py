"""
Review queue: send humans only the messages where AI labels are informative.

Within each held-out split (`test`, `enrich_test`):

- ai_directional — every message any AI source labelled bullish / bearish
  (this includes every disagreement between sources: two sources can only
  disagree if at least one of them is non-neutral). Taken in full.
- ai_neutral     — messages every AI source called neutral. Only a random
  control sample is taken; it measures how many real bull / bear messages
  the AI misses.

Because the strata are sampled at different rates, raw agreement on the
queue would overstate how often stance appears. `stratum_weights` gives each
human-labelled item weight = stratum population / labelled items in that
stratum, so metrics computed with these weights estimate the full split.
"""
import random
from collections import Counter, defaultdict
from typing import Optional

HELD_OUT = ("test", "enrich_test")


def build_queue(
    samples: list[dict], sources: dict[str, dict[str, dict]], control: int = 60,
    splits: tuple[str, ...] = HELD_OUT, seed: int = 7, keep: frozenset = frozenset(),
) -> list[dict]:
    """
    `sources` maps a source name to its {id: label} dict. Each message is judged
    by whichever sources labelled it (e.g. v1 only covers `test`); messages no
    source labelled are skipped. `control` neutral messages are spread over
    splits in proportion to their neutral counts.

    `keep` are ids a human already labelled from a uniformly random draw of the
    split; they join their stratum on top of the control sample. That keeps each
    stratum a simple random sample, so the weights stay valid and no earlier
    human work is wasted.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in samples:
        if s["split"] not in splits:
            continue
        stances = {name: src[s["id"]]["stance"] for name, src in sources.items() if s["id"] in src}
        if not stances:
            continue
        directional = any(v != "neutral" for v in stances.values())
        strata[(s["split"], "ai_directional" if directional else "ai_neutral")].append(
            {"id": s["id"], "split": s["split"], "disputed": len(set(stances.values())) > 1,
             "ai_stances": stances}
        )

    neutral_total = sum(len(v) for (_, stratum), v in strata.items() if stratum == "ai_neutral")
    queue = []
    for (split, stratum), items in sorted(strata.items()):
        population = len(items)
        if stratum == "ai_neutral":
            k = round(control * population / neutral_total) if neutral_total else 0
            kept = [it for it in items if it["id"] in keep]
            rest = [it for it in items if it["id"] not in keep]
            items = kept + rng.sample(rest, min(k, len(rest)))
        for it in items:
            queue.append({**it, "stratum": stratum, "stratum_population": population})

    rng.shuffle(queue)  # no runs of "all directional" that would cue the annotator
    return queue


def stratum_weights(queue: list[dict], labelled_ids: set[str]) -> dict[str, float]:
    """id → weight for labelled queue items; each stratum's weights sum to its population."""
    labelled = [q for q in queue if q["id"] in labelled_ids]
    counts = Counter((q["split"], q["stratum"]) for q in labelled)
    return {q["id"]: q["stratum_population"] / counts[(q["split"], q["stratum"])] for q in labelled}


def summarize(queue: list[dict]) -> dict:
    out: dict[str, dict] = defaultdict(dict)
    for (split, stratum), n in Counter((q["split"], q["stratum"]) for q in queue).items():
        pop = next(q["stratum_population"] for q in queue if q["split"] == split and q["stratum"] == stratum)
        out[split][stratum] = {"queued": n, "population": pop}
    for split in out:
        out[split]["disputed"] = sum(1 for q in queue if q["split"] == split and q["disputed"])
    return dict(out)


def queue_ids_in_order(queue: list[dict], split: Optional[str] = None) -> list[str]:
    return [q["id"] for q in queue if split in (None, q["split"])]
