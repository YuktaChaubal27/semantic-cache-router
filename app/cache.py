"""Semantic cache.

Version, tenant and TTL are applied as filters *inside* the vector search,
so a stale or foreign entry can never shadow a valid one that sits just
behind it in similarity order.
"""
import json
import time
import uuid
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, Range

from .config import settings
from . import verify
from .guard import compatible
from .vectorstore import client, ensure_collection

SHARED = "__shared__"


@dataclass
class CacheHit:
    score: float                       # embedding similarity
    payload: dict
    verify_score: float | None = None  # cross-encoder duplicate probability


def _collection() -> str:
    name = settings()["cache"]["collection"]
    ensure_collection(name)
    return name


def _filter(version: int, tenant: str | None) -> Filter:
    ttl = settings()["cache"]["ttl_seconds"]
    return Filter(must=[
        FieldCondition(key="version", match=MatchValue(value=version)),
        FieldCondition(key="tenant", match=MatchValue(value=tenant or SHARED)),
        FieldCondition(key="created_at", range=Range(gte=time.time() - ttl)),
    ])


def nearest(vec: list[float], query: str, version: int, tenant: str | None = None) -> CacheHit | None:
    """Three checks, cheapest first:
    1. embedding similarity >= threshold (inside the vector search)
    2. guard: numbers and acronyms must not conflict
    3. verifier: a cross-encoder must agree the questions are duplicates
    """
    cfg = settings()["cache"]
    res = client().query_points(
        _collection(), query=vec, query_filter=_filter(version, tenant), limit=5,
        score_threshold=cfg["threshold"], with_payload=True,
    ).points
    for p in res:  # already sorted by similarity
        cached_q = p.payload["query"]
        if not compatible(query, cached_q):
            continue
        if not verify.enabled():
            return CacheHit(score=p.score, payload=json.loads(p.payload["result"]))
        prob = verify.duplicate_prob(query, cached_q)
        if prob >= cfg["verify_threshold"]:
            return CacheHit(score=p.score, payload=json.loads(p.payload["result"]), verify_score=prob)
    return None


def store(vec: list[float], query: str, result: dict, version: int, tenant: str | None = None) -> None:
    client().upsert(_collection(), points=[PointStruct(
        id=str(uuid.uuid4()),
        vector=vec,
        payload={
            "query": query,
            "result": json.dumps(result),
            "version": version,
            "tenant": tenant or SHARED,
            "created_at": time.time(),
        },
    )])
