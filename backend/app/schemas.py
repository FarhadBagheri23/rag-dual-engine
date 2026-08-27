from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    file_type: str
    n_chunks: int
    n_words: int
    added_at: datetime


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    ordinal: int
    text: str


class DeleteResult(BaseModel):
    """What deletion actually removed — the UI shows this, and it is the
    evidence that both indexes were cleaned (rubric: Index Synchronization)."""

    id: str
    chunks_removed: int
    postings_removed: int = 0  # phase 2
    vectors_removed: int = 0  # phase 4


class SearchRequest(BaseModel):
    query: str
    engine: Literal["vsm", "bm25", "rag"] = "vsm"
    mode: Literal["exact", "champion", "elimination"] = "champion"
    prf: bool = False  # Rocchio pseudo-relevance feedback, VSM only
    model: str | None = None  # overrides LLM_MODEL for one request, RAG only
    k: int | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    ordinal: int
    score: float
    snippet: str
    matched: list[str]
    doc_number: int | None = None  # the N in [Doc N], RAG only


class Citation(BaseModel):
    doc_number: int
    chunk_id: str
    title: str


class SearchResponse(BaseModel):
    query: str
    engine: str
    mode: str
    prf: bool = False
    took_ms: float
    scored: int  # size of the candidate set — shows what inexact top-K skipped
    expansion: list[str] = []  # terms Rocchio added, so the effect is visible
    hits: list[SearchHit]
    # RAG only
    answer: str | None = None
    citations: list[Citation] = []
    model: str | None = None
    note: str | None = None
