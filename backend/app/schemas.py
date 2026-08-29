from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.security import MAX_PASSWORD_BYTES


class Credentials(BaseModel):
    """Login and registration take the same pair.

    ponytail: a pattern, not pydantic's EmailStr — that pulls in the
    email-validator package to RFC-check an address nothing is ever sent to.
    Here an email is a login identifier, so the only useful check is that it
    looks like one and the user has not fat-fingered it.

    The password ceiling is real, though: bcrypt hashes at most 72 bytes and
    raises above that, so bounding it here turns a 500 into a 422 that names
    the limit.
    """

    email: str = Field(
        min_length=5, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)


class UserOut(BaseModel):
    """Never contains password_hash. That column leaves the database layer
    only on the login path, and never leaves the process."""

    id: str
    email: str
    role: Literal["client", "admin"]
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str  # 30 minutes
    refresh_token: str  # 7 days
    token_type: str = "bearer"
    user: UserOut


class RefreshRequest(BaseModel):
    refresh_token: str


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # Free-form per-turn render state: citations and hits for RAG, mode and prf
    # for the lexical engines. Opaque to the backend, which only stores it.
    meta: dict[str, Any] | None = None


class ConversationOut(BaseModel):
    """A sidebar row. `engine` is what renders the VSM / BM25 / RAG tag."""

    id: str
    engine: Literal["vsm", "bm25", "rag"]
    title: str
    n_messages: int = 0
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[ConversationMessage] = []


class ConversationSave(BaseModel):
    id: str | None = None  # absent or unknown starts a new thread
    engine: Literal["vsm", "bm25", "rag"]
    messages: list[ConversationMessage]


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    file_type: str
    n_chunks: int
    n_words: int
    added_at: datetime


class DeleteResult(BaseModel):
    """What deletion actually removed — the UI shows this, and it is the
    evidence that both indexes were cleaned (rubric: Index Synchronization)."""

    id: str
    chunks_removed: int
    postings_removed: int = 0  # phase 2
    vectors_removed: int = 0  # phase 4


class Turn(BaseModel):
    """One prior message in the conversation, for RAG query rewriting."""

    role: Literal["user", "assistant"]
    content: str


class SearchRequest(BaseModel):
    query: str
    engine: Literal["vsm", "bm25", "rag"] = "vsm"
    mode: Literal["exact", "champion", "elimination"] = "champion"
    prf: bool = False  # Rocchio pseudo-relevance feedback, VSM only
    model: str | None = None  # overrides LLM_MODEL for one request, RAG only
    k: int | None = None
    # Widen retrieval past the corpus, RAG only. Off by default: it costs an
    # outbound request, and most questions about an uploaded corpus should be
    # answered from that corpus.
    web: bool = False
    # Earlier turns, oldest first. RAG only: a follow-up like "just it?" cannot
    # be embedded into anything meaningful on its own (RAG slides s24).
    history: list[Turn] = []


class SearchHit(BaseModel):
    """One row of a ranked list.

    `doc_id` is the ranked unit for VSM and BM25 — the score belongs to the
    document (spec §3.3.2), and a document appears at most once. `chunk_id` says
    only which chunk the snippet was taken from. For RAG the two are inverted in
    importance: the chunk is what was retrieved and cited, and `doc_id` is just
    the file it came out of.
    """

    doc_id: str
    chunk_id: str  # the snippet's source, not the ranked unit
    title: str
    score: float
    snippet: str
    matched: list[str]
    doc_number: int | None = None  # the N in [Doc N], RAG only
    # This document's column of the term × document weight matrix (slide
    # 7-Scoring s29): what each query term contributed to `score`. The values
    # sum to `score`, which is what makes the heatmap checkable rather than
    # decorative. Empty for RAG — dense retrieval has no per-term decomposition.
    contrib: dict[str, float] = {}


class Citation(BaseModel):
    doc_number: int
    # A web passage has no chunk in the corpus and a corpus passage has no URL,
    # so exactly one of these is set. Which one is what tells the UI whether to
    # render a document or an outbound link.
    chunk_id: str | None = None
    title: str
    url: str | None = None


class WebSource(BaseModel):
    """One web result that went into the prompt, cited or not.

    Kept apart from `hits` because the two are not the same kind of thing: a
    hit is a chunk of the user's own corpus with a similarity score, and this
    is a page on the internet with an address. Merging them would mean a score
    field that is meaningless for half the rows.
    """

    doc_number: int
    title: str
    url: str
    domain: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    engine: str
    mode: str
    prf: bool = False
    took_ms: float
    scored: int  # documents actually scored — shows what inexact top-K skipped
    # Per-retrieval breakdown of `scored`. Empty for a single pass; two entries
    # when Rocchio ran, because it retrieves twice and the total can then exceed
    # the collection size.
    passes: list[int] = []
    expansion: list[str] = []  # terms Rocchio added, so the effect is visible
    hits: list[SearchHit]
    # RAG only
    answer: str | None = None
    citations: list[Citation] = []
    model: str | None = None
    note: str | None = None
    coverage: list[str] = []  # corpus topics, shown when a query is rejected
    web: list[WebSource] = []  # web passages placed in the prompt, RAG only
    rewritten: str | None = None  # the standalone question actually retrieved for
    # Heatmap rows: the terms actually ranked on, heaviest first. Not derivable
    # from `hits` — a term that matched nothing appears in no column and would
    # silently vanish, when "this term found nothing" is exactly what the reader
    # needs to see.
    terms: list[str] = []
