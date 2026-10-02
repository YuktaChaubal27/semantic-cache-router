"""Second-stage duplicate check for cache candidates.

Embeddings capture topic but not direction, so "cancel my subscription" and
"restart my subscription" look nearly identical. The verifier reads both
questions together and decides whether one cached answer fits both.

Two kinds, chosen by cache.verifier in config.yaml:
  "llm:<litellm model>"   a small LLM answers YES/NO (costs a few tokens)
  "<cross-encoder name>"  a local sentence-transformers CrossEncoder (free)
Every check returns (score 0..1, cost_usd) so verification cost is logged.
"""
from functools import lru_cache

from .config import settings

LLM_PROMPT = (
    "You check a cache for a customer-support assistant. Two users asked the "
    "questions below. Would one correct answer fully and accurately answer BOTH "
    "questions? Treat them as different if they ask for a different action, "
    "about a different thing, or with any different specific detail. "
    "Reply with exactly one word: YES or NO."
)


def _name() -> str | None:
    return settings()["cache"].get("verifier")


def enabled() -> bool:
    return bool(_name())


def is_llm() -> bool:
    return (_name() or "").startswith("llm:")


@lru_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_name())


def check(a: str, b: str) -> tuple[float, float]:
    """Returns (duplicate score 0..1, cost in USD)."""
    if is_llm():
        from .llm import complete
        out = complete(_name()[4:], [
            {"role": "system", "content": LLM_PROMPT},
            {"role": "user", "content": f"Question A: {a}\nQuestion B: {b}"},
        ], max_tokens=3)
        score = 1.0 if out["text"].strip().upper().startswith("YES") else 0.0
        return score, out["cost_usd"]
    return float(_cross_encoder().predict([(a, b)])[0]), 0.0
