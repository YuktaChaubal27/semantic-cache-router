"""Sweep the cache threshold over hand-written paraphrase and near-miss pairs.

A pair counts as a cache hit only if similarity >= threshold AND the guard
allows it (numbers/acronyms don't conflict), exactly as in production.
Picks the lowest threshold with zero false hits on near-miss pairs.

    python scripts/tune_threshold.py data/threshold_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from app.embeddings import embed
from app.guard import compatible
from app.normalize import normalize


def cos(a, b):  # embeddings are already unit-normalized
    return sum(x * y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--lo", type=float, default=0.50)
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    for p in pairs:
        p["sim"] = cos(embed(normalize(p["a"])), embed(normalize(p["b"])))
        p["guard_ok"] = compatible(p["a"], p["b"])
    pos = [p for p in pairs if p["should_hit"]]
    neg = [p for p in pairs if not p["should_hit"]]

    def hits(group, t):
        return [p for p in group if p["sim"] >= t and p["guard_ok"]]

    print(f"{len(pos)} paraphrase pairs, {len(neg)} near-miss pairs\n")
    print(f"{'threshold':>9}  {'hit rate':>8}  {'false hits':>10}")
    rows, chosen = [], None
    n = int(round((args.hi - args.lo) / args.step)) + 1
    for i in range(n):
        t = round(args.lo + i * args.step, 4)
        hit_rate = len(hits(pos, t)) / max(len(pos), 1)
        false_hits = len(hits(neg, t))
        rows.append({"threshold": t, "hit_rate": hit_rate, "false_hits": false_hits})
        if chosen is None and false_hits == 0:
            chosen = rows[-1]
    # Print every 5th row plus the chosen one, so the table stays readable.
    for r in rows:
        if round(r["threshold"] * 100) % 5 == 0 or r is chosen:
            mark = "  <- chosen" if r is chosen else ""
            print(f"{r['threshold']:>9.2f}  {r['hit_rate']:>8.0%}  {r['false_hits']:>10}{mark}")

    print("\nParaphrases (should hit), lowest first:")
    for p in sorted(pos, key=lambda p: p["sim"]):
        flag = "" if p["guard_ok"] else "   [guard blocks]"
        print(f"  {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}{flag}")

    print("\nNear-misses (must not hit), highest first:")
    for p in sorted(neg, key=lambda p: -p["sim"]):
        flag = "   [guard blocks]" if not p["guard_ok"] else ""
        print(f"  {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}{flag}")

    if chosen:
        print(f"\nChosen threshold: {chosen['threshold']:.2f} "
              f"(hit rate {chosen['hit_rate']:.0%}, 0 false hits). Put it in config.yaml.")
    else:
        print("\nNo threshold in range had zero false hits.")

    Path("results").mkdir(exist_ok=True)
    json.dump({"sweep": rows, "chosen": chosen}, open("results/threshold.json", "w"), indent=2)


if __name__ == "__main__":
    main()
