"""Exact-match guard for details that embeddings tend to ignore.

Embedding models score "refund after 30 days" and "refund after 14 days"
as near-identical. If both queries carry specifics (numbers, acronyms like
EU/US) and those specifics differ, the cached answer is not safe to reuse.
"""
import re

_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")    # "30", "14", "2.5" but not the 2 in "2FA"
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")        # "EU", "US", "SMS"


def key_terms(query: str) -> set[str]:
    return set(_NUMBER.findall(query)) | set(_ACRONYM.findall(query))


def compatible(a: str, b: str) -> bool:
    """False only when both queries have specifics and they conflict."""
    ka, kb = key_terms(a), key_terms(b)
    return not (ka and kb and ka != kb)
