"""Answer checks used for escalation and for deciding what is safe to cache."""
import re

_CITATION = re.compile(r"\[\d+\]")
_IDK = re.compile(r"\b(i don'?t know|i do not know|cannot (?:find|answer)|not (?:in|mentioned in) the (?:sources|context))\b", re.I)


def is_idk(answer: str) -> bool:
    return bool(_IDK.search(answer))


def has_citation(answer: str) -> bool:
    return bool(_CITATION.search(answer))


def needs_escalation(result: dict, min_top_score: float, require_citation: bool) -> bool:
    answer = (result.get("answer") or "").strip()
    scores = result.get("retrieval_scores") or [0.0]
    return (
        not answer
        or is_idk(answer)
        or (require_citation and not has_citation(answer))
        or max(scores) < min_top_score
    )


def is_cacheable(result: dict, require_citation: bool) -> bool:
    answer = (result.get("answer") or "").strip()
    return (
        not result.get("error")
        and bool(answer)
        and not is_idk(answer)
        and (has_citation(answer) or not require_citation)
    )


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present. Cheap, deterministic eval metric."""
    if not expected_keywords:
        return 1.0
    a = answer.lower()
    return sum(k.lower() in a for k in expected_keywords) / len(expected_keywords)
