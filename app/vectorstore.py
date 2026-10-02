"""One shared local Qdrant client (on-disk mode, no server required)."""
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import settings
from .embeddings import dim


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(path=settings()["paths"]["qdrant"])


def ensure_collection(name: str) -> None:
    c = client()
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=VectorParams(size=dim(), distance=Distance.COSINE))
