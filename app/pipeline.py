"""The request flow: cache -> retrieve -> route -> generate -> maybe escalate."""
import time

from . import cache, db
from .config import settings
from .embeddings import embed
from .normalize import normalize
from .quality import is_cacheable, needs_escalation
from .rag import generate, retrieve
from .router import classify


def answer(query: str, tenant: str | None = None, mode: str = "full", run_id: str | None = None) -> dict:
    """mode='baseline' skips cache and routing and always uses the large model."""
    cfg = settings()
    small, large = cfg["models"]["small"], cfg["models"]["large"]
    esc = cfg["escalation"]
    t0 = time.perf_counter()
    base = {"query": query, "tenant": tenant, "mode": mode, "run_id": run_id}

    def ms() -> float:
        return (time.perf_counter() - t0) * 1000

    try:
        version = db.get_corpus_version()
        q_vec = None

        if mode == "full":
            q_vec = embed(normalize(query))
            hit = cache.nearest(q_vec, query, version=version, tenant=tenant)
            if hit:  # threshold, guard and verifier are all applied inside cache.nearest
                rid = db.log_request(**base, cache_hit=1, latency_ms=ms())
                return {**hit.payload, "cache_hit": True, "cache_score": hit.score,
                        "verify_score": hit.verify_score, "request_id": rid}

        hits = retrieve(query)
        route = "complex" if mode == "baseline" else classify(query, [h["score"] for h in hits])
        result = generate(query, hits, model=large if route == "complex" else small)
        total_cost, in_tok, out_tok = result["cost_usd"], result["input_tokens"], result["output_tokens"]

        escalated = False
        if route == "simple" and needs_escalation(result, esc["min_top_score"], esc["require_citation"]):
            escalated = True
            result = generate(query, hits, model=large)
            total_cost += result["cost_usd"]
            in_tok += result["input_tokens"]
            out_tok += result["output_tokens"]

        if mode == "full" and is_cacheable(result, esc["require_citation"]):
            payload = {k: result[k] for k in ("answer", "sources", "model")}
            cache.store(q_vec, query, payload, version=version, tenant=tenant)

        rid = db.log_request(
            **base, cache_hit=0, route=route, escalated=int(escalated), model=result["model"],
            input_tokens=in_tok, output_tokens=out_tok, cost_usd=total_cost, latency_ms=ms(),
        )
        return {
            "answer": result["answer"], "sources": result["sources"], "model": result["model"],
            "cache_hit": False, "route": route, "escalated": escalated,
            "cost_usd": total_cost, "request_id": rid,
        }
    except Exception as e:  # log failures too, so error rate shows up on the dashboard
        rid = db.log_request(**base, cache_hit=0, latency_ms=ms(), error=repr(e))
        raise
