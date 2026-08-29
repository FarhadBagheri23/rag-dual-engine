import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.auth import get_current_user
from app.core.exceptions import ProviderError
from app.database import documents as db
from app.engines.lexical.index import index
from app.engines.lexical.text import snippet, tokenize
from app.engines.registry import ENGINES, SUPPORTS_PRF
from app.engines.semantic import rag
from app.schemas import SearchRequest, SearchResponse

# Both routes need a signed-in user and neither needs the row, so the guard
# goes on the router — one line, and a route added later inherits it instead of
# shipping unauthenticated because someone forgot the decorator.
router = APIRouter(
    prefix="/api/search", tags=["search"], dependencies=[Depends(get_current_user)]
)


def _render(chunk: dict, doc_id: str, r: dict, query_terms: set[str]) -> dict:
    """One hit as the UI needs it: title, snippet, highlighted terms.

    `doc_id` is the ranked unit and `chunk["id"]` is only where the snippet was
    taken from — the two are separate fields because they answer different
    questions, and collapsing them is what made a document appear twice in one
    ranked list.
    """
    return {
        "doc_id": doc_id,
        "chunk_id": chunk["id"],
        "title": chunk["title"],
        "score": round(r["score"], 6),
        "snippet": snippet(chunk["text"], query_terms),
        # Most informative first, and drop terms with zero idf: a term
        # occurring in every document contributes nothing to the score, so
        # highlighting it is noise ("a" lit up in every snippet).
        "matched": sorted(
            (
                t
                for t in r["matched"] & set(index.forward.get(chunk["id"], {}))
                if index.idf(t) > 0
            ),
            key=index.idf,
            reverse=True,
        ),
        "doc_number": r.get("doc_number"),
    }


def _enrich(hits: list[dict], query: str) -> list[dict]:
    """Lexical hits — one row per *document*, per spec §3.3.2.

    Each row still shows a snippet, taken from whichever chunk of that document
    best matches the query. Chunk text lives in SQLite, the source of truth —
    the index stores only what it needs to rank, never a second copy.
    """
    texts = {c["id"]: c for c in db.all_chunks()}
    query_terms = set(tokenize(query))

    out = []
    for r in hits:
        cid = index.best_chunk(r["doc_id"], query_terms)
        chunk = texts.get(cid)
        if chunk is None:  # index ahead of the db; a rebuild will reconcile
            continue
        out.append(_render(chunk, r["doc_id"], r, query_terms))
    return out


def _enrich_chunks(hits: list[dict], query: str) -> list[dict]:
    """RAG hits — one row per *chunk*, because that is what was retrieved and
    what went into the prompt. A citation has to point at the passage the claim
    came from, not at a whole PDF."""
    texts = {c["id"]: c for c in db.all_chunks()}
    query_terms = set(tokenize(query))

    out = []
    for r in hits:
        chunk = texts.get(r["chunk_id"])
        if chunk is None:
            continue
        out.append(_render(chunk, chunk["doc_id"], r, query_terms))
    return out


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
    result = engine(
        req.query,
        k=req.k,
        mode=req.mode,
        prf=req.prf,
        model=req.model,
        history=req.history,
    )
    took_ms = (time.perf_counter() - started) * 1000

    return {
        "query": req.query,
        "engine": req.engine,
        "mode": req.mode,
        "prf": req.prf,
        "took_ms": round(took_ms, 2),
        "scored": result["scored"],
        "passes": result.get("passes", []),
        "expansion": result.get("expansion", []),
        # RAG ranks the chunks it retrieved; the lexical engines rank documents.
        "hits": (_enrich_chunks if req.engine == "rag" else _enrich)(
            result["hits"], req.query
        ),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "model": result.get("model"),
        "note": result.get("note"),
        "coverage": result.get("coverage", []),
        "rewritten": result.get("rewritten"),
    }


@router.post("/stream")
def search_stream(req: SearchRequest):
    """RAG only, as newline-delimited JSON — one event per line.

    NDJSON rather than Server-Sent Events because EventSource cannot POST, and
    the query belongs in a body. `fetch` + a stream reader handles this in a
    dozen lines of browser code, with no protocol to hand-roll.

    Errors arrive as an `error` line, not a status code: by the time generation
    fails the response has already started and the headers are long gone.
    """
    if req.engine != "rag":
        raise HTTPException(
            400, "Streaming is only meaningful for 'rag' — the lexical engines "
            "return in under a millisecond. Use POST /api/search."
        )

    def events():
        started = time.perf_counter()
        try:
            for event in rag.stream(
                req.query, k=req.k, model=req.model, history=req.history
            ):
                if "stage" in event and "hits" in event:
                    event = {**event, "hits": _enrich_chunks(event["hits"], req.query)}
                elif "done" in event:
                    event = {
                        "done": {
                            **event["done"],
                            "hits": _enrich_chunks(event["done"]["hits"], req.query),
                            "took_ms": round((time.perf_counter() - started) * 1000, 2),
                        }
                    }
                yield json.dumps(event) + "\n"
        except ProviderError as exc:
            yield json.dumps({"error": exc.detail}) + "\n"
        except Exception as exc:  # noqa: BLE001 — the browser needs *something*
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        # nginx and friends buffer by default, which would defeat the point.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
