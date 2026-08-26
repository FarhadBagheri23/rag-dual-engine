from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, search
from app.core.config import settings
from app.database.session import init_db
from app.engines.lexical.index import INDEX_PATH, index


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not INDEX_PATH.exists():
        # The index is derived from SQLite, so a missing or deleted index file
        # is recoverable rather than fatal.
        index.rebuild_from_db()
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


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.llm_api_key),
        "llm_model": settings.llm_model,
    }
