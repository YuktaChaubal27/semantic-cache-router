"""Replay the eval set plus paraphrases through baseline and full systems.

Paraphrases simulate realistic repeat traffic; without them the cache's
value is invisible.

    python scripts/benchmark.py data/eval_set.jsonl --reset-cache
"""
import argparse
import json
import random
import time
import uuid
from pathlib import Path

import _path  # noqa: F401
from app import db
from app.config import settings
from app.pipeline import answer
from app.quality import keyword_score
from app.vectorstore import client


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(mode, traffic, run_id):
    for item, q in traffic:
        try:
            r = answer(q, mode=mode, run_id=run_id)
            db.set_quality(r["request_id"], keyword_score(r["answer"], item.get("keywords", [])))
        except Exception as e:
            print(f"  [{mode}] error on {q!r}: {e}")
    with db.connect() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM requests WHERE run_id = ?", (run_id,))]
    ok = [r for r in rows if not r["error"]]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "total_cost": sum(r["cost_usd"] for r in ok),
        "cost_per_request": sum(r["cost_usd"] for r in ok) / max(len(ok), 1),
        "p50_ms": pct([r["latency_ms"] for r in ok], 0.50),
        "p95_ms": pct([r["latency_ms"] for r in ok], 0.95),
        "quality": sum(r["quality"] or 0 for r in ok) / max(len(ok), 1),
        "cache_hit_rate": sum(r["cache_hit"] for r in ok) / max(len(ok), 1),
        "simple_share": sum(r["route"] == "simple" for r in ok) / max(len(ok), 1),
        "escalation_rate": sum(r["escalated"] for r in ok) / max(sum(r["route"] == "simple" for r in ok), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set")
    ap.add_argument("--reset-cache", action="store_true", help="start the full run with an empty cache")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.eval_set) if l.strip()]
    originals = [(it, it["question"]) for it in items]
    repeats = [(it, p) for it in items for p in it.get("paraphrases", [])]
    random.Random(args.seed).shuffle(repeats)
    traffic = originals + repeats  # first-time questions, then repeat traffic

    if args.reset_cache:
        name = settings()["cache"]["collection"]
        if client().collection_exists(name):
            client().delete_collection(name)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Replaying {len(traffic)} requests ({len(originals)} unique, {len(repeats)} paraphrased)")
    base = run("baseline", traffic, f"baseline-{stamp}-{uuid.uuid4().hex[:6]}")
    full = run("full", traffic, f"full-{stamp}-{uuid.uuid4().hex[:6]}")

    def change(a, b):
        return (b - a) / a * 100 if a else 0.0

    summary = {
        "baseline": base, "full": full,
        "cost_change_pct": change(base["total_cost"], full["total_cost"]),
        "p95_change_pct": change(base["p95_ms"], full["p95_ms"]),
        "quality_delta": full["quality"] - base["quality"],
        "eval_questions": len(items),
    }
    Path("results").mkdir(exist_ok=True)
    json.dump(summary, open(f"results/benchmark-{stamp}.json", "w"), indent=2)

    print(f"\n{'':<18}{'baseline':>12}{'full':>12}")
    for k in ("cost_per_request", "p50_ms", "p95_ms", "quality", "cache_hit_rate", "simple_share", "escalation_rate"):
        print(f"{k:<18}{base[k]:>12.4f}{full[k]:>12.4f}")
    print(f"\nCost change: {summary['cost_change_pct']:+.1f}%   p95 change: {summary['p95_change_pct']:+.1f}%   "
          f"quality delta: {summary['quality_delta']:+.3f} on {len(items)} questions")


if __name__ == "__main__":
    main()
