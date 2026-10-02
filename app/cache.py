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
from .guard import compatible
from .vectorstore import client, ensure_collection

SHARED = "__shared__"


@dataclass
class CacheHit:
    score: float
    payload: dict


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
    """Best cached entry whose specifics (numbers, acronyms) don't conflict with the query."""
    res = client().query_points(
        _collection(), query=vec, query_filter=_filter(version, tenant), limit=5, with_payload=True
    ).points
    for p in res:  # already sorted by similarity
        if compatible(query, p.payload["query"]):
            return CacheHit(score=p.score, payload=json.loads(p.payload["result"]))
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
