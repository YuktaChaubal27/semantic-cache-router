from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import answer

app = FastAPI(title="Semantic cache + cost-aware router")


class Ask(BaseModel):
    query: str
    tenant_id: str | None = None  # set this when answers are personalized


@app.post("/ask")
def ask(body: Ask):
    try:
        return answer(body.query, tenant=body.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
def health():
    return {"ok": True}
