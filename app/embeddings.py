from functools import lru_cache

from .config import settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings()["embedding_model"])


def embed(text: str) -> list[float]:
    # normalize_embeddings=True makes dot product equal cosine similarity
    return _model().encode(text, normalize_embeddings=True).tolist()


def dim() -> int:
    m = _model()
    # newer sentence-transformers renamed this method
    get = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return get()
