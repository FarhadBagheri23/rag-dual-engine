"""Register, log in, refresh — and the two dependencies everything else uses.

The dependencies live here rather than in a deps.py: they are the only thing
this module exports besides its routes, and a file with two functions in it is
not worth the import.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.security import create_token, decode_token, verify_password
from app.database import users as db
from app.schemas import Credentials, RefreshRequest, TokenPair, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])

# auto_error=False so a missing header reaches get_current_user and comes back
# as our own 401 with a WWW-Authenticate challenge, rather than HTTPBearer's
# bare 403 — which is the wrong status and tells the client nothing.
bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    """The account behind a valid, unexpired access token.

    The row is re-read every request instead of trusting the role in the
    payload. A token stays validly signed for its full 30 minutes, so without
    this a deleted account keeps working and a demoted admin keeps admin.
    """
    if creds is None:
        raise _unauthorized("Not authenticated")
    payload = decode_token(creds.credentials, kind="access")
    if payload is None:
        raise _unauthorized("Invalid or expired token")
    user = db.by_id(payload["sub"])
    if user is None:
        raise _unauthorized("Account no longer exists")
    return user


def get_current_admin(user: dict = Depends(get_current_user)) -> dict:
    """403, not 401: the caller authenticated fine, they just are not an admin.
    Returning 401 here would send the frontend to the login screen for a user
    who is already correctly logged in."""
    if user["role"] != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "This action requires an admin account"
        )
    return user


def _issue(user: dict) -> dict:
    return {
        "access_token": create_token(user["id"], user["role"], "access"),
        "refresh_token": create_token(user["id"], user["role"], "refresh"),
        "user": user,
    }


@router.post("/register", response_model=TokenPair, status_code=201)
def register(body: Credentials):
    """Always creates a 'client'. There is deliberately no way to ask for a
    role here — the admin is seeded from the environment at startup, so no
    request body can talk this endpoint into minting one."""
    user = db.create(body.email, body.password)
    if user is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "That email is already registered")
    return _issue(user)


@router.post("/login", response_model=TokenPair)
def login(body: Credentials):
    user = db.by_email(body.email)
    # One message for both "no such user" and "wrong password", so the endpoint
    # cannot be used to enumerate which emails have accounts. verify_password
    # still runs against a real hash when the user exists, and returns early
    # when it does not — the timing difference is not worth a dummy hash here.
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return _issue({k: v for k, v in user.items() if k != "password_hash"})


@router.post("/refresh", response_model=TokenPair)
def refresh(body: RefreshRequest):
    """Trade a 7-day refresh token for a fresh 30-minute access token.

    ponytail: no refresh-token table, and the old token stays usable until it
    expires. Storing them would buy revocation, which needs a logout that
    reaches the server and a lookup on every refresh — worth it for a bank, not
    for a corpus search. Rotating JWT_SECRET is the global sign-out.
    """
    payload = decode_token(body.refresh_token, kind="refresh")
    if payload is None:
        raise _unauthorized("Invalid or expired refresh token")
    user = db.by_id(payload["sub"])
    if user is None:
        raise _unauthorized("Account no longer exists")
    return _issue(user)


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    """Who the current token belongs to — the frontend calls this on reload to
    decide whether a stored token is still good before rendering anything."""
    return user


@router.get("/config")
def auth_config():
    """Token lifetimes, so the client can schedule a refresh instead of
    discovering expiry as a failed request mid-search."""
    return {
        "access_token_minutes": settings.access_token_minutes,
        "refresh_token_days": settings.refresh_token_days,
    }
