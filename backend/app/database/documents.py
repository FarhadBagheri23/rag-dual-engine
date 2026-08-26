"""Every SQL statement in the project lives here."""

from datetime import datetime, timezone

from app.database.session import connect


def insert(doc: dict, chunks: list[dict]):
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, title, filename, file_type, n_words, added_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc["id"],
                doc["title"],
                doc["filename"],
                doc["file_type"],
                doc["n_words"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.executemany(
            "INSERT INTO chunks (id, doc_id, ordinal, text) VALUES (?, ?, ?, ?)",
            [(c["id"], doc["id"], c["ordinal"], c["text"]) for c in chunks],
        )


def list_all() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT d.*, COUNT(c.id) AS n_chunks FROM documents d"
            " LEFT JOIN chunks c ON c.doc_id = d.id"
            " GROUP BY d.id ORDER BY d.added_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get(doc_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def chunk_ids(doc_id: str) -> list[str]:
    """The chunk ids the other indexes keyed their entries on — read this
    *before* deleting, so deletion can purge them everywhere."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
        ).fetchall()
    return [r["id"] for r in rows]


def all_chunks() -> list[dict]:
    """Every chunk in the corpus — used to (re)build the derived indexes."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.*, d.title FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " ORDER BY c.doc_id, c.ordinal"
        ).fetchall()
    return [dict(r) for r in rows]


def delete(doc_id: str) -> int:
    """Delete a document and its chunks. Returns the chunk count removed."""
    with connect() as conn:
        n = conn.execute(
            "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
        ).rowcount
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    return n
