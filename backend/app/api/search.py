import time

from fastapi import APIRouter, HTTPException

from app.database import documents as db
from app.engines.lexical.index import index
from app.engines.lexical.text import snippet, tokenize
from app.engines.registry import ENGINES
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest):
    engine = ENGINES.get(req.engine)
    if engine is None:
        raise HTTPException(400, f"Unknown engine '{req.engine}'")

    started = time.perf_counter()
    result = engine(req.query, k=req.k, mode=req.mode)
    took_ms = (time.perf_counter() - started) * 1000

    # Chunk text lives in SQLite, the source of truth — the index stores only
    # what it needs to rank, never a second copy of the text.
    texts = {c["id"]: c for c in db.all_chunks()}
    query_terms = set(tokenize(req.query))

    hits = []
    for r in result["hits"]:
        chunk = texts.get(r["chunk_id"])
        if chunk is None:  # index ahead of the db; a rebuild will reconcile
            continue
        hits.append(
            {
                "chunk_id": r["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "ordinal": chunk["ordinal"],
                "score": round(r["score"], 6),
                "snippet": snippet(chunk["text"], query_terms),
                "matched": sorted(r["matched"] & set(index.forward.get(r["chunk_id"], {}))),
            }
        )

    return {
        "query": req.query,
        "engine": req.engine,
        "mode": req.mode,
        "took_ms": round(took_ms, 2),
        "scored": result["scored"],
        "hits": hits,
    }
