"""Conversation SQL. Every statement takes a user_id.

Ownership is a WHERE clause on each query rather than a check in the route,
because a route that forgets the check returns someone else's history and no
test notices. Here, a query that forgets it does not compile into anything
useful — there is no function to call that reads a conversation without saying
whose it is.
"""

import json
import uuid
from datetime import datetime, timezone

from app.database.session import connect

TITLE_MAX = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_for_user(user_id: str) -> list[dict]:
    """The sidebar: newest-touched first, with a message count per thread."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.engine, c.title, c.created_at, c.updated_at,"
            "       COUNT(m.id) AS n_messages"
            " FROM conversations c LEFT JOIN messages m ON m.conv_id = c.id"
            " WHERE c.user_id = ?"
            " GROUP BY c.id ORDER BY c.updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get(conv_id: str, user_id: str) -> dict | None:
    """One thread with its messages, or None if it is not this user's.

    Missing and not-yours collapse to the same answer on purpose: telling a
    caller "that exists but is not yours" leaks that it exists.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT role, content, meta FROM messages WHERE conv_id = ?"
            " ORDER BY ordinal",
            (conv_id,),
        ).fetchall()
    return {
        **dict(row),
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "meta": json.loads(m["meta"]) if m["meta"] else None,
            }
            for m in messages
        ],
    }


def save(user_id: str, conv_id: str | None, engine: str, messages: list[dict]) -> str:
    """Create or replace one thread wholesale. Returns its id.

    ponytail: the client PUTs the whole transcript and the messages are deleted
    and re-inserted, rather than an append endpoint that has to agree with the
    client about ordinals. A thread is tens of rows and the write is one
    transaction, so the simpler contract costs nothing. Switch to append-only
    if transcripts ever get long enough for the rewrite to show up in a trace.
    """
    title = (messages[0]["content"] if messages else "").strip()[:TITLE_MAX] or "Untitled"
    now = _now()

    with connect() as conn:
        if conv_id and conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        ).fetchone():
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conv_id),
            )
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        else:
            # Unknown id, or one belonging to someone else — either way this
            # user gets a fresh thread rather than writing into that row.
            conv_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO conversations (id, user_id, engine, title,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, user_id, engine, title, now, now),
            )

        conn.executemany(
            "INSERT INTO messages (id, conv_id, role, content, meta, ordinal,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    uuid.uuid4().hex[:12],
                    conv_id,
                    m["role"],
                    m["content"],
                    json.dumps(m["meta"]) if m.get("meta") else None,
                    i,
                    now,
                )
                for i, m in enumerate(messages)
            ],
        )
    return conv_id


def delete(conv_id: str, user_id: str) -> bool:
    """True if a thread of this user's was removed. Messages go by CASCADE."""
    with connect() as conn:
        n = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conv_id, user_id),
        ).rowcount
    return n > 0
