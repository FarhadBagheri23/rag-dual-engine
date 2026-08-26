from datetime import datetime

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
