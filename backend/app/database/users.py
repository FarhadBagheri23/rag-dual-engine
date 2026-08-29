"""User SQL. Same shape as documents.py: every statement about users is here."""

import sqlite3
import uuid
from datetime import datetime, timezone

from app.core.security import hash_password
from app.database.session import connect


def create(email: str, password: str, role: str = "client") -> dict | None:
    """Insert one account. None when the email is already taken.

    The UNIQUE index is what decides that, not a prior SELECT: two requests can
    both pass a check-then-insert, only one can survive the constraint.
    """
    user = {
        "id": uuid.uuid4().hex[:12],
        "email": email.strip(),
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    user["id"],
                    user["email"],
                    hash_password(password),
                    role,
                    user["created_at"],
                ),
            )
    except sqlite3.IntegrityError:
        return None
    return user


def by_email(email: str) -> dict | None:
    """Includes password_hash — only the login path should call this."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip(),)
        ).fetchone()
    return dict(row) if row else None


def by_id(user_id: str) -> dict | None:
    """The row behind a token's `sub`.

    Read on every authenticated request rather than trusting the role baked
    into the token: a demoted or deleted user keeps a valid signature for up to
    30 minutes, and this is what makes that window close immediately.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT id, email, role, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def seed_admin(email: str, password: str) -> bool:
    """Create the configured admin if that email has no account. Idempotent."""
    if not email or not password or by_email(email):
        return False
    return create(email, password, role="admin") is not None
