"""Tune the cache on hand-written paraphrase and near-miss pairs.

Part A: embedding similarity + guard only (one threshold).
Part B: two-stage. The embedding threshold becomes a loose floor that finds
        candidates, and a cross-encoder verifier makes the final call.
Each part picks the lowest setting with zero false hits on near-miss pairs.

    python scripts/tune_threshold.py data/threshold_pairs.jsonl
    python scripts/tune_threshold.py data/threshold_pairs.jsonl --floor 0.55
"""
import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from app.config import settings
from app.embeddings import embed
from app.guard import compatible
from app.normalize import normalize


def cos(a, b):  # embeddings are already unit-normalized
    return sum(x * y for x, y in zip(a, b))


def sweep(pos, neg, values, passes):
    rows, chosen = [], None
    for v in values:
        hit_rate = sum(passes(p, v) for p in pos) / max(len(pos), 1)
        false_hits = sum(passes(p, v) for p in neg)
        rows.append({"value": v, "hit_rate": hit_rate, "false_hits": false_hits})
        if chosen is None and false_hits == 0:
            chosen = rows[-1]
    return rows, chosen


def show(rows, chosen, label, every=5):
    print(f"{label:>9}  {'hit rate':>8}  {'false hits':>10}")
    for r in rows:
        if round(r["value"] * 100) % every == 0 or r is chosen:
            mark = "  <- chosen" if r is chosen else ""
            print(f"{r['value']:>9.2f}  {r['hit_rate']:>8.0%}  {r['false_hits']:>10}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--floor", type=float, default=0.50, help="embedding floor for the two-stage check")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    for p in pairs:
        p["sim"] = cos(embed(normalize(p["a"])), embed(normalize(p["b"])))
        p["guard_ok"] = compatible(p["a"], p["b"])
    pos = [p for p in pairs if p["should_hit"]]
    neg = [p for p in pairs if not p["should_hit"]]
    print(f"{len(pos)} paraphrase pairs, {len(neg)} near-miss pairs")
    result = {}

    # ---- Part A: embedding + guard ----
    print("\n=== A. Embedding similarity + guard ===")
    values = [round(0.40 + i * 0.01, 2) for i in range(58)]  # 0.40 .. 0.97
    rows, chosen_a = sweep(pos, neg, values, lambda p, t: p["sim"] >= t and p["guard_ok"])
    show(rows, chosen_a, "threshold")
    result["embedding_only"] = {"sweep": rows, "chosen": chosen_a}

    print("\nHardest near-misses for embeddings (not blocked by the guard):")
    for p in sorted([p for p in neg if p["guard_ok"]], key=lambda p: -p["sim"])[:5]:
        print(f"  {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")

    # ---- Part B: two-stage with cross-encoder verifier ----
    if not settings()["cache"].get("verifier"):
        print("\nNo verifier set in config.yaml; skipping part B.")
    else:
        from app.verify import duplicate_probs
        probs = duplicate_probs([(p["a"], p["b"]) for p in pairs])
        for p, pr in zip(pairs, probs):
            p["prob"] = pr

        below = [p for p in pos if p["sim"] < args.floor]
        print(f"\n=== B. Two-stage: embedding floor {args.floor:.2f} + guard + verifier ===")
        print(f"({len(below)} of {len(pos)} paraphrases fall below the floor and can never hit)")
        values = [round(0.05 + i * 0.05, 2) for i in range(19)]  # 0.05 .. 0.95
        rows, chosen_b = sweep(pos, neg, values,
                               lambda p, v: p["sim"] >= args.floor and p["guard_ok"] and p["prob"] >= v)
        show(rows, chosen_b, "verifier", every=5)
        result["two_stage"] = {"floor": args.floor, "sweep": rows, "chosen": chosen_b}

        print("\nNear-misses the verifier rates most duplicate-like (must stay low):")
        for p in sorted([p for p in neg if p["guard_ok"] and p["sim"] >= args.floor], key=lambda p: -p["prob"])[:5]:
            print(f"  verifier {p['prob']:.3f}  sim {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")
        print("\nParaphrases the verifier rates least duplicate-like (should stay high):")
        for p in sorted(pos, key=lambda p: p["prob"])[:5]:
            print(f"  verifier {p['prob']:.3f}  sim {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")

    # ---- Summary ----
    print("\n=== Summary (zero false hits) ===")
    if chosen_a:
        print(f"  Embedding only: threshold {chosen_a['value']:.2f} -> hit rate {chosen_a['hit_rate']:.0%}")
    if result.get("two_stage", {}).get("chosen"):
        c = result["two_stage"]["chosen"]
        print(f"  Two-stage:      floor {args.floor:.2f}, verifier {c['value']:.2f} -> hit rate {c['hit_rate']:.0%}")
        print(f"\n  Put in config.yaml:  threshold: {args.floor:.2f}   verify_threshold: {c['value']:.2f}")
    elif "two_stage" in result:
        print("  Two-stage: no verifier setting had zero false hits. Try a higher --floor.")

    Path("results").mkdir(exist_ok=True)
    json.dump(result, open("results/threshold.json", "w"), indent=2)


if __name__ == "__main__":
    main()
