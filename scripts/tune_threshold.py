"""Tune the cache on hand-written paraphrase and near-miss pairs.

Part A: embedding similarity + guard only (one threshold).
Part B: two-stage. The embedding threshold becomes a loose floor that finds
        candidates, and a verifier (small LLM or cross-encoder) makes the final call.
Each part picks the lowest setting with zero false hits on near-miss pairs.

    python scripts/tune_threshold.py data/threshold_pairs.jsonl
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

    # ---- Part B: two-stage with verifier ----
    if not settings()["cache"].get("verifier"):
        print("\nNo verifier set in config.yaml; skipping part B.")
    else:
        from app import verify
        vt = settings()["cache"]["verify_threshold"]
        print(f"\nRunning the verifier on all {len(pairs)} pairs...")
        total_cost = 0.0
        for p in pairs:
            p["vscore"], c = verify.check(p["a"], p["b"])
            total_cost += c
            p["verified"] = p["vscore"] >= vt

        print(f"\n=== B. Two-stage: embedding floor + guard + verifier ({settings()['cache']['verifier']}) ===")
        floors = [round(0.30 + i * 0.05, 2) for i in range(12)]  # 0.30 .. 0.85
        rows, chosen_b = sweep(pos, neg, floors,
                               lambda p, f: p["sim"] >= f and p["guard_ok"] and p["verified"])
        # Share of paraphrase checks the verifier would be asked to make at each floor
        show(rows, chosen_b, "floor")
        result["two_stage"] = {"verifier": settings()["cache"]["verifier"], "sweep": rows, "chosen": chosen_b,
                               "verifier_cost_per_check": total_cost / max(len(pairs), 1)}

        wrong_yes = [p for p in neg if p["guard_ok"] and p["verified"]]
        wrong_no = [p for p in pos if not p["verified"]]
        print(f"\nVerifier mistakes: {len(wrong_yes)} near-misses accepted, {len(wrong_no)} paraphrases rejected")
        for p in sorted(wrong_yes, key=lambda p: -p["sim"]):
            print(f"  accepted (wrong)  sim {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")
        for p in sorted(wrong_no, key=lambda p: -p["sim"]):
            print(f"  rejected (wrong)  sim {p['sim']:.3f}  {p['a']!r} vs {p['b']!r}")
        print(f"\nVerifier cost: ${total_cost:.4f} for {len(pairs)} checks "
              f"(${total_cost / max(len(pairs), 1):.6f} per check)")

    # ---- Summary ----
    print("\n=== Summary (zero false hits) ===")
    if chosen_a:
        print(f"  Embedding only: threshold {chosen_a['value']:.2f} -> hit rate {chosen_a['hit_rate']:.0%}")
    if result.get("two_stage", {}).get("chosen"):
        c = result["two_stage"]["chosen"]
        print(f"  Two-stage:      floor {c['value']:.2f} + verifier -> hit rate {c['hit_rate']:.0%}")
        print(f"\n  Put in config.yaml:  threshold: {c['value']:.2f}   (keep the verifier line)")
    elif "two_stage" in result:
        print("  Two-stage: the verifier accepted a near-miss at every floor (see mistakes above).")

    Path("results").mkdir(exist_ok=True)
    json.dump(result, open("results/threshold.json", "w"), indent=2)


if __name__ == "__main__":
    main()
