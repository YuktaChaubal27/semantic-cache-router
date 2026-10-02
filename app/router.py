"""Rule-based router, plus an optional tiny-LLM classifier to compare against.

Pure functions with no heavy imports, so they are cheap to unit test.
"""
import re
from statistics import pstdev

COMPLEX_WORDS = {
    "compare", "comparison", "versus", "vs", "why", "explain", "difference",
    "differences", "tradeoff", "tradeoffs", "pros", "cons", "analyze", "impact",
    "should", "best", "recommend", "how come",
}
_WORD = re.compile(r"[A-Za-z0-9']+")


def features(query: str, retrieval_scores: list[float] | None = None) -> dict:
    words = _WORD.findall(query)
    lower = {w.lower() for w in words}
    # Crude entity count: capitalized words not at the start, plus numbers.
    entities = sum(1 for i, w in enumerate(words) if (i > 0 and w[0].isupper()) or w.isdigit())
    scores = retrieval_scores or []
    return {
        "n_words": len(words),
        "n_entities": entities,
        "complex_words": len(lower & COMPLEX_WORDS) + (" and " in f" {query.lower()} "),
        "n_questions": query.count("?"),
        "top_score": max(scores) if scores else 0.0,
        "score_spread": pstdev(scores) if len(scores) > 1 else 0.0,
    }


def classify(query: str, retrieval_scores: list[float] | None = None) -> str:
    f = features(query, retrieval_scores)
    points = 0
    points += f["n_words"] > 18
    points += f["n_entities"] >= 3
    points += 2 * min(f["complex_words"], 1)
    points += f["n_questions"] > 1
    # Flat, low scores mean no single chunk answers it: likely needs synthesis.
    if retrieval_scores:
        points += f["top_score"] < 0.5 and f["score_spread"] < 0.05
    return "complex" if points >= 2 else "simple"


LLM_ROUTER_PROMPT = (
    "Classify the user question as 'simple' (a single fact lookup answerable from "
    "one passage) or 'complex' (needs comparison, reasoning, or several passages). "
    "Reply with one word: simple or complex."
)


def classify_llm(query: str, model: str) -> tuple[str, float]:
    """Returns (route, cost_usd). Used by scripts/eval_router.py for comparison."""
    from .llm import complete
    out = complete(model, [
        {"role": "system", "content": LLM_ROUTER_PROMPT},
        {"role": "user", "content": query},
    ], max_tokens=3)
    route = "complex" if "complex" in out["text"].lower() else "simple"
    return route, out["cost_usd"]
