"""Unit tests for the pure logic. No models, network or vector DB needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.normalize import normalize
from app.quality import has_citation, is_cacheable, is_idk, keyword_score, needs_escalation
from app.router import classify


def test_normalize():
    assert normalize("  How do I RESET my password?! ") == "how do i reset my password"


def test_router_simple_and_complex():
    assert classify("How do I reset my password?") == "simple"
    assert classify("Compare the refund policy for EU and US customers") == "complex"
    assert classify("Why is SMS not supported?") == "complex"


def test_flat_low_retrieval_scores_tip_a_borderline_query():
    q = "What is the rule in Germany, France, Spain?"  # one point from entities alone
    assert classify(q) == "simple"
    assert classify(q, [0.41, 0.40, 0.40, 0.39]) == "complex"   # no chunk clearly answers it
    assert classify(q, [0.82, 0.55, 0.40, 0.31]) == "simple"    # one chunk clearly does


def test_idk_and_citation():
    assert is_idk("I don't know.")
    assert not is_idk("Open Settings [1].")
    assert has_citation("Open Settings [1].")
    assert not has_citation("Open Settings.")


def test_escalation():
    good = {"answer": "Open Settings [1].", "retrieval_scores": [0.8]}
    assert not needs_escalation(good, 0.35, True)
    assert needs_escalation({"answer": "I don't know", "retrieval_scores": [0.8]}, 0.35, True)
    assert needs_escalation({"answer": "Open Settings.", "retrieval_scores": [0.8]}, 0.35, True)
    assert needs_escalation({"answer": "Open Settings [1].", "retrieval_scores": [0.2]}, 0.35, True)
    assert needs_escalation({"answer": "", "retrieval_scores": [0.9]}, 0.35, True)


def test_never_cache_errors_empty_or_idk():
    assert is_cacheable({"answer": "Open Settings [1]."}, True)
    assert not is_cacheable({"answer": "Open Settings [1].", "error": "boom"}, True)
    assert not is_cacheable({"answer": "   "}, True)
    assert not is_cacheable({"answer": "I don't know"}, True)
    assert not is_cacheable({"answer": "Open Settings."}, True)


def test_keyword_score():
    assert keyword_score("Refunds within 14 days", ["14 days", "30 days"]) == 0.5
    assert keyword_score("anything", []) == 1.0


def test_guard_blocks_conflicting_specifics():
    from app.guard import compatible
    assert not compatible("refund policy in the EU", "refund policy in the US")
    assert not compatible("can I get a refund after 30 days", "can I get a refund after 14 days")
    assert compatible("how do I reset my password", "password reset steps")       # no specifics
    assert compatible("can I turn on 2FA", "how to enable two-factor authentication")  # one side only
    assert compatible("EU refund deadline", "refund window for EU customers")       # same specifics
