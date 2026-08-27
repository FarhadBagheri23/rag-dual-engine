from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, models, search
from app.core.config import settings
from app.database import documents as db
from app.database.session import init_db
from app.engines.lexical.index import INDEX_PATH, index
from app.engines.semantic import vectordb


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Both indexes are derived from SQLite, so drift is repairable rather than
    # fatal: reconcile each against the corpus instead of trusting it.
    n_chunks = len(db.all_chunks())
    if not INDEX_PATH.exists() or index.n_chunks != n_chunks:
        index.rebuild_from_db()
    if vectordb.count() != n_chunks or vectordb.stale():
        vectordb.rebuild_from_db()
    yield


app = FastAPI(
    title="Dual-Engine Document Search & RAG", version="0.2.0", lifespan=lifespan
)

# ponytail: wide-open CORS is fine for a local dev/demo app; lock to the real
# origin if this is ever deployed somewhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(models.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.llm_api_key),
        "llm_model": settings.llm_model,
    }
