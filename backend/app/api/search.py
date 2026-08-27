import time

from fastapi import APIRouter, HTTPException

from app.database import documents as db
from app.engines.lexical.index import index
from app.engines.lexical.text import snippet, tokenize
from app.engines.registry import ENGINES, SUPPORTS_PRF
from app.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest):
    engine = ENGINES.get(req.engine)
    if engine is None:
        raise HTTPException(400, f"Unknown engine '{req.engine}'")
    if req.prf and req.engine not in SUPPORTS_PRF:
        raise HTTPException(
            400,
            f"Pseudo-relevance feedback is defined on the vector space model; "
            f"'{req.engine}' does not support it. Supported: {sorted(SUPPORTS_PRF)}",
        )

    started = time.perf_counter()
    result = engine(req.query, k=req.k, mode=req.mode, prf=req.prf, model=req.model)
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
                "doc_number": r.get("doc_number"),
            }
        )

    return {
        "query": req.query,
        "engine": req.engine,
        "mode": req.mode,
        "prf": req.prf,
        "took_ms": round(took_ms, 2),
        "scored": result["scored"],
        "expansion": result.get("expansion", []),
        "hits": hits,
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "model": result.get("model"),
        "note": result.get("note"),
    }
