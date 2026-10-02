"""Compare the rule-based router against a tiny LLM classifier on hand labels.

    python scripts/eval_router.py data/router_labels.jsonl [--llm]
"""
import argparse
import json

import _path  # noqa: F401
from app.config import settings
from app.router import classify, classify_llm


def report(name, preds, labels, cost=0.0):
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    # "complex labelled simple" is the costly error: it hurts quality
    under = sum(p == "simple" and l == "complex" for p, l in zip(preds, labels))
    over = sum(p == "complex" and l == "simple" for p, l in zip(preds, labels))
    print(f"{name:<12} accuracy {acc:.0%}  under-routed {under}  over-routed {over}  cost ${cost:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM classifier")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labels) if l.strip()]
    labels = [r["label"] for r in rows]
    # Rules are evaluated on query text alone here; in production they also see retrieval scores.
    report("rules", [classify(r["query"]) for r in rows], labels)

    if args.llm:
        model = settings()["models"]["small"]
        preds, cost = [], 0.0
        for r in rows:
            p, c = classify_llm(r["query"], model)
            preds.append(p)
            cost += c
        report("llm", preds, labels, cost)


if __name__ == "__main__":
    main()
