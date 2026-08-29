"""ChromaDB — the dense counterpart to the inverted index (RAG slides s34).

Same contract as InvertedIndex: add(chunks, doc_id) and remove(chunk_ids),
both called only from services/corpus.py. Chunk ids are shared between the
two stores, which is what makes synchronized deletion a lookup rather than a
search.

Embeddings are supplied explicitly rather than letting Chroma call its own
default embedding function, so the model in config.py is the only thing that
decides how text becomes a vector.
"""

import logging

import chromadb

from app.core.config import settings
from app.engines.semantic import embedder

COLLECTION = "chunks"

# chromadb 0.6.3 calls posthog's pre-6.x capture() signature, so every store
# operation logs "capture() takes 1 positional argument but 3 were given".
# Harmless, but it drowns the log. Disabling telemetry in Settings does not
# stop it — the client is constructed before the setting is read.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def _client():
    return chromadb.PersistentClient(
        path=str(settings.index_dir / "chroma"),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def collection():
    # cosine, to match the similarity the course teaches — Chroma's default is
    # squared L2, which ranks differently for non-unit vectors.
    #
    # The embedding model is recorded here because vectors are only comparable
    # to others from the same model. Swapping models leaves the chunk *count*
    # identical, so a count-based reconcile would not notice, and the store
    # would keep answering with silently meaningless vectors.
    return _client().get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model},
    )


def stale() -> bool:
    """True when the stored vectors did not come from the current model.

    Missing metadata counts as stale: get_or_create_collection only applies
    metadata at creation time, so an existing collection keeps whatever it was
    built with. Treating None as "unknown, therefore rebuild" is the safe
    reading — the alternative is comparing query vectors from one model against
    document vectors from another, which returns confident nonsense rather than
    an error.
    """
    return (collection().metadata or {}).get("embed_model") != settings.embed_model


def add(chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    collection().add(
        ids=[c["id"] for c in chunks],
        embeddings=embedder.embed_documents(texts),
        documents=texts,
        metadatas=[{"doc_id": doc_id, "ordinal": c["ordinal"]} for c in chunks],
    )
    return len(chunks)


def remove(chunk_ids: list[str]) -> int:
    if not chunk_ids:
        return 0
    col = collection()
    existing = set(col.get(ids=chunk_ids, include=[])["ids"])
    if existing:
        col.delete(ids=list(existing))
    return len(existing)


def query(text: str, k: int) -> list[dict]:
    """Nearest chunks by cosine similarity to the query embedding."""
    return query_vector(embedder.embed_query(text), k)


def query_vector(vector: list[float], k: int) -> list[dict]:
    """The search half alone, for callers that already embedded the query.

    rag.stream needs the two steps separated so it can report "embedding" and
    "retrieving" as they actually happen — a progress indicator that groups
    them is guessing at where the time went.
    """
    col = collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["distances"],
    )
    # Chroma returns cosine *distance* in [0, 2]; report similarity instead so
    # a bigger number means a better match, as in every other engine here.
    return [
        {"chunk_id": cid, "score": 1.0 - dist}
        for cid, dist in zip(res["ids"][0], res["distances"][0])
    ]


def count() -> int:
    return collection().count()


def rebuild_from_db() -> int:
    """Re-derive every vector from SQLite, the source of truth.

    Needed whenever the dense store falls behind the corpus: a collection
    deleted by hand, a changed embedding model, or — the case that found this —
    documents indexed before the vector store existed at all. Without it, an
    older corpus stays permanently invisible to semantic search.
    """
    from collections import defaultdict

    from app.database import documents as db

    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # nothing to drop on a first run

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in db.all_chunks():
        by_doc[chunk["doc_id"]].append(chunk)

    return sum(add(chunks, doc_id) for doc_id, chunks in by_doc.items())
