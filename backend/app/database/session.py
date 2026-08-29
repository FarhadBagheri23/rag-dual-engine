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

-- Accounts live beside the corpus rather than in a second database file.
-- Chroma and inverted.pkl are *derived* stores, rebuildable from these tables;
-- a user row is not rebuildable from anything, so it belongs in the source of
-- truth. One file to back up, and foreign keys that actually resolve.
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    -- NOCASE so Alice@x and alice@x are one account, and UNIQUE so two
    -- simultaneous registrations cannot both win. Both enforced by SQLite,
    -- which no amount of application-level checking does as reliably.
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'client'
                  CHECK (role IN ('client', 'admin')),
    created_at    TEXT NOT NULL
);

-- One search or chat session. Engine is stored per conversation, not per
-- message, because switching engines starts a new thread in the UI.
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    engine     TEXT NOT NULL CHECK (engine IN ('vsm', 'bm25', 'rag')),
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- The sidebar's only query: this user's threads, newest touched first.
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    conv_id    TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    -- Whatever the UI needs to re-render the turn: citations, hits, mode, prf.
    -- JSON in one TEXT column because it is only ever read back whole and
    -- handed to the client — nothing filters or joins on it, so columns would
    -- buy nothing and a migration every time the UI changes shape.
    meta       TEXT,
    ordinal    INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, ordinal);
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
