"""SQLite is the source of truth for document text.

Both retrieval indexes are *derived* from it: the inverted index (phase 2) and
the vector store (phase 4) hold only what can be rebuilt from these tables.
That is what makes a rebuild possible and a desync detectable.
"""

import sqlite3
from contextlib import contextmanager

from app.core.config import settings

DB_PATH = settings.index_dir / "meta.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    filename   TEXT NOT NULL,
    file_type  TEXT NOT NULL,
    n_words    INTEGER NOT NULL,
    added_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id      TEXT PRIMARY KEY,
    doc_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@contextmanager
def connect():
    """One connection per operation, committed on clean exit.

    ponytail: no pool, no ORM, no session middleware. SQLite on a local disk
    with one writer does not need them; revisit only if this ever fronts a
    real concurrent workload.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # off by default; CASCADE needs it
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
