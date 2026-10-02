"""Sweep the cache threshold over hand-written paraphrase and near-miss pairs.

Picks the lowest threshold with zero false hits on near-miss pairs, and
reports how many true paraphrases still hit at that value.

    python scripts/tune_threshold.py data/threshold_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from app.embeddings import embed
from app.normalize import normalize


def cos(a, b):  # embeddings are already unit-normalized
    return sum(x * y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--lo", type=float, default=0.85)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    for p in pairs:
        p["sim"] = cos(embed(normalize(p["a"])), embed(normalize(p["b"])))
    pos = [p for p in pairs if p["should_hit"]]
    neg = [p for p in pairs if not p["should_hit"]]
    print(f"{len(pos)} paraphrase pairs, {len(neg)} near-miss pairs\n")
    print(f"{'threshold':>9}  {'hit rate':>8}  {'false hits':>10}")

    rows, chosen = [], None
    n = int(round((args.hi - args.lo) / args.step)) + 1
    for i in range(n):
        t = round(args.lo + i * args.step, 4)
        hit_rate = sum(p["sim"] >= t for p in pos) / max(len(pos), 1)
        false_hits = sum(p["sim"] >= t for p in neg)
        rows.append({"threshold": t, "hit_rate": hit_rate, "false_hits": false_hits})
        print(f"{t:>9.2f}  {hit_rate:>8.0%}  {false_hits:>10}")
        if chosen is None and false_hits == 0:
            chosen = rows[-1]

    worst = sorted(neg, key=lambda p: -p["sim"])[:5]
    print("\nClosest near-misses (these set your floor):")
    for p in worst:
        print(f"  {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")

    if chosen:
        print(f"\nChosen threshold: {chosen['threshold']:.2f} "
              f"(hit rate {chosen['hit_rate']:.0%}, 0 false hits). Put it in config.yaml.")
    else:
        print("\nNo threshold in range had zero false hits. Raise --hi or add tenant/locale filters.")

    Path("results").mkdir(exist_ok=True)
    json.dump({"sweep": rows, "chosen": chosen}, open("results/threshold.json", "w"), indent=2)


if __name__ == "__main__":
    main()
