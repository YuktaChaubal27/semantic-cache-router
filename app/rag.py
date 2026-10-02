"""Minimal RAG pipeline. Retrieval and generation are split so the router
can look at retrieval scores before choosing a model.

If you already have a RAG baseline, replace retrieve() and generate()
with calls into it and keep the return shapes the same.
"""
from qdrant_client.models import PointStruct

from .config import settings
from .embeddings import embed
from .llm import complete
from .vectorstore import client, ensure_collection

SYSTEM = (
    "Answer the question using only the numbered sources. Cite sources inline "
    "like [1]. If the sources do not contain the answer, reply exactly: I don't know."
)


def _collection() -> str:
    name = settings()["retrieval"]["collection"]
    ensure_collection(name)
    return name


def retrieve(query: str) -> list[dict]:
    pts = client().query_points(
        _collection(), query=embed(query), limit=settings()["retrieval"]["top_k"], with_payload=True
    ).points
    return [{"score": p.score, "text": p.payload["text"], "source": p.payload["source"]} for p in pts]


def generate(query: str, hits: list[dict], model: str) -> dict:
    context = "\n\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    out = complete(model, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"},
    ])
    out["answer"] = out.pop("text").strip()
    out["sources"] = list(dict.fromkeys(h["source"] for h in hits))  # unique, in rank order
    out["retrieval_scores"] = [h["score"] for h in hits]
    return out


def index_chunks(chunks: list[dict]) -> None:
    """chunks: [{"id": str, "text": str, "source": str}]"""
    import uuid
    name = _collection()
    client().upsert(name, points=[
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])), vector=embed(c["text"]),
                    payload={"text": c["text"], "source": c["source"]})
        for c in chunks
    ])
