"""One shared local Qdrant client (on-disk mode, no server required)."""
import atexit
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import settings
from .embeddings import dim


@lru_cache
def client() -> QdrantClient:
    c = QdrantClient(path=settings()["paths"]["qdrant"])
    atexit.register(c.close)  # close cleanly before Python shuts down
    return c


def ensure_collection(name: str) -> None:
    c = client()
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=VectorParams(size=dim(), distance=Distance.COSINE))
