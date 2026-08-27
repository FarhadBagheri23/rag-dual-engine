"""The single writer.

Every mutation of the corpus goes through `add` or `remove`. Nothing else in
the codebase writes to a store, so the derived indexes cannot drift from the
source of truth. Phase 2 adds the inverted index to both functions; phase 4
adds the vector store. They are the only two lines that will ever be added.
"""

import shutil
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import DocumentNotFound
from app.database import documents as db
from app.engines.lexical.index import index as lexical_index
from app.engines.semantic import vectordb
from app.services import ingest


def add(upload_path: Path, filename: str) -> dict:
    """Parse, chunk and persist one file. Returns the document row."""
    text, title = ingest.parse(upload_path)
    doc_id = ingest.new_doc_id()
    chunks = ingest.chunk(text, doc_id)

    # Keep the original bytes under an id-based name: user-supplied filenames
    # are never used as paths, so "../../etc/passwd" cannot escape the dir.
    stored = settings.upload_dir / f"{doc_id}{upload_path.suffix.lower()}"
    shutil.move(upload_path, stored)

    doc = {
        "id": doc_id,
        "title": title,
        "filename": filename,
        "file_type": upload_path.suffix.lower().lstrip("."),
        "n_words": len(text.split()),
    }
    db.insert(doc, chunks)
    lexical_index.add(chunks, doc_id)
    lexical_index.save()
    vectordb.add(chunks, doc_id)
    return {**doc, "n_chunks": len(chunks), "added_at": db.get(doc_id)["added_at"]}


def remove(doc_id: str) -> dict:
    """Delete a document from every store. Returns what each one removed."""
    doc = db.get(doc_id)
    if doc is None:
        raise DocumentNotFound(doc_id)

    ids = db.chunk_ids(doc_id)  # read before deleting — the other stores key on these
    postings_removed = lexical_index.remove(ids)
    lexical_index.save()
    vectors_removed = vectordb.remove(ids)
    chunks_removed = db.delete(doc_id)

    stored = settings.upload_dir / f"{doc_id}.{doc['file_type']}"
    stored.unlink(missing_ok=True)

    return {
        "id": doc_id,
        "chunks_removed": chunks_removed,
        "postings_removed": postings_removed,
        "vectors_removed": vectors_removed,
    }


def listing() -> list[dict]:
    return db.list_all()
