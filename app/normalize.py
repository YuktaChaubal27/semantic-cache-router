import re
import string

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_SPACE = re.compile(r"\s+")


def normalize(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    q = _PUNCT.sub(" ", query.lower())
    return _SPACE.sub(" ", q).strip()
