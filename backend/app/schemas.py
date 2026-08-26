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
    engine: Literal["vsm"] = "vsm"  # bm25 and rag join in phases 3 and 4
    mode: Literal["exact", "champion", "elimination"] = "champion"
    k: int | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    ordinal: int
    score: float
    snippet: str
    matched: list[str]


class SearchResponse(BaseModel):
    query: str
    engine: str
    mode: str
    took_ms: float
    scored: int  # size of the candidate set — shows what inexact top-K skipped
    hits: list[SearchHit]
