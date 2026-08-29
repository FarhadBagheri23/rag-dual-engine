"""Password hashing and JWT minting. No session state anywhere.

ponytail: bcrypt and PyJWT called directly. passlib's CryptContext is an
abstraction over the same bcrypt call, for swapping between hash schemes this
project will never swap between; fastapi-users would be more configuration
than this file is code.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings

# bcrypt hashes at most 72 bytes and raises above that (5.x stopped truncating
# silently). Rejected at the schema instead, so the caller gets a 422 naming
# the limit rather than a 500 from in here.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """bcrypt with a per-password random salt.

    gensalt() draws 16 fresh random bytes every call, and hashpw embeds them in
    the returned string ($2b$12$<22-char salt><31-char digest>). So the salt is
    stored with the hash, no separate column, and two users who pick the same
    password still get different hashes — which is the point: one leaked table
    cannot be attacked with a single precomputed rainbow table.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time compare; the salt is read back out of `password_hash`."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False  # malformed hash in the row — treat as a failed login


def create_token(user_id: str, role: str, kind: str) -> str:
    """Sign an access (30 min) or refresh (7 day) token.

    `kind` is inside the payload, not just implied by lifetime, so a refresh
    token cannot be presented as an access token. Without it the two are
    indistinguishable to the verifier and the 30-minute limit becomes 7 days.
    """
    lifetime = (
        timedelta(minutes=settings.access_token_minutes)
        if kind == "access"
        else timedelta(days=settings.refresh_token_days)
    )
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "role": role, "kind": kind, "iat": now, "exp": now + lifetime},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str, kind: str) -> dict | None:
    """Payload if the signature, expiry and kind all check out, else None.

    `algorithms` is pinned to the configured one rather than read from the
    token's own header — trusting the header is how "alg: none" gets you.
    """
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.InvalidTokenError:  # bad signature, expired, malformed
        return None
    return payload if payload.get("kind") == kind else None
