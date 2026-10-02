"""Second-stage duplicate check with a cross-encoder.

Embeddings (a bi-encoder) encode each question separately, so they capture
topic but not direction: "cancel my subscription" and "restart my
subscription" look nearly identical. A cross-encoder reads both questions
together and judges whether they ask the same thing. It is slower, so it
only runs on candidates the embedding search already found.
"""
from functools import lru_cache

from .config import settings


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings()["cache"]["verifier"])


def enabled() -> bool:
    return bool(settings()["cache"].get("verifier"))


def duplicate_prob(a: str, b: str) -> float:
    """Probability (0 to 1) that a and b ask the same question."""
    return float(_model().predict([(a, b)])[0])


def duplicate_probs(pairs: list[tuple[str, str]]) -> list[float]:
    return [float(x) for x in _model().predict(pairs)]
