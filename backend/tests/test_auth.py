"""Phase 6 check: hashing, token kinds, role gates, and history ownership.

The interesting cases here are the ones where a *wrong* answer still looks like
a working app: a refresh token accepted as an access token, a client reaching an
admin route, one user reading another's search history.

Run with:  cd backend && python -m tests.test_auth
"""

import atexit
import uuid

from fastapi.testclient import TestClient

from app.core import security
from app.database import conversations as convdb
from app.database import users as userdb
from app.database.session import init_db, connect
from app.main import app

# These checks run against the real dev database, like every other test here.
# Unlike an index rebuild, though, accounts *accumulate* — so each one this file
# creates is registered and dropped on exit. Conversations go with them by
# CASCADE. atexit rather than a __main__ block, so it also fires under pytest.
_CREATED: list[str] = []


def _email() -> str:
    address = f"{uuid.uuid4().hex[:10]}@example.com"
    _CREATED.append(address)
    return address


@atexit.register
def _drop_test_accounts():
    if not _CREATED:
        return
    with connect() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executemany(
            "DELETE FROM users WHERE email = ?", [(e,) for e in _CREATED]
        )


def test_hash_is_salted_and_verifiable():
    pw = "correct-horse-battery"
    a, b = security.hash_password(pw), security.hash_password(pw)

    assert pw not in a, "plaintext survived into the hash"
    assert a != b, "same password hashed identically — the salt is not random"
    assert a.startswith("$2b$"), f"not a bcrypt hash: {a[:8]}"
    # Both verify despite differing, because each carries its own salt.
    assert security.verify_password(pw, a)
    assert security.verify_password(pw, b)
    assert not security.verify_password("wrong", a)
    assert not security.verify_password(pw, "not-a-hash"), "malformed hash accepted"
    print("  distinct salts, both verify, wrong password rejected")


def test_a_refresh_token_is_not_an_access_token():
    """The 7-day token must not open a door the 30-minute one guards, or the
    access lifetime is decorative."""
    access = security.create_token("u1", "client", "access")
    refresh = security.create_token("u1", "client", "refresh")

    assert security.decode_token(access, "access")["sub"] == "u1"
    assert security.decode_token(refresh, "refresh")["sub"] == "u1"
    assert security.decode_token(refresh, "access") is None, "refresh used as access"
    assert security.decode_token(access, "refresh") is None, "access used as refresh"
    # Flip a character in the signature.
    assert security.decode_token(access[:-2] + ("aa" if access[-2:] != "aa" else "bb"),
                                 "access") is None, "tampered signature accepted"
    assert security.decode_token("garbage", "access") is None
    print("  kinds do not cross, tampered and malformed tokens rejected")


def test_register_never_produces_an_admin():
    init_db()
    client = TestClient(app)
    email = _email()

    # Even when the body explicitly asks for one.
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123", "role": "admin"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["user"]["role"] == "client", "registration minted an admin"

    dupe = client.post(
        "/api/auth/register", json={"email": email.upper(), "password": "password123"}
    )
    assert dupe.status_code == 409, f"case-variant duplicate allowed: {dupe.status_code}"
    print("  role=admin in the body ignored, case-variant duplicate rejected")


def test_role_gates_on_document_routes():
    init_db()
    client = TestClient(app)

    anon = client.get("/api/documents")
    assert anon.status_code == 401, f"corpus listed without a token: {anon.status_code}"

    email = _email()
    tokens = client.post(
        "/api/auth/register", json={"email": email, "password": "password123"}
    ).json()
    as_client = {"Authorization": f"Bearer {tokens['access_token']}"}

    # A client reads the corpus (the search page's doc counter needs it)...
    assert client.get("/api/documents", headers=as_client).status_code == 200
    # ...but cannot change it. 403, not 401 — they are logged in, just not admin.
    assert client.delete("/api/documents/nope", headers=as_client).status_code == 403
    assert client.post("/api/documents", headers=as_client).status_code == 403

    admin = userdb.create(_email(), "password123", role="admin")
    as_admin = {
        "Authorization": f"Bearer {security.create_token(admin['id'], 'admin', 'access')}"
    }
    # The admin gets past the gate and on to the real 404.
    assert client.delete("/api/documents/nope", headers=as_admin).status_code == 404
    print("  anon 401, client 403, admin through to the handler")


def test_history_is_not_readable_across_accounts():
    init_db()
    alice = userdb.create(_email(), "password123")
    bob = userdb.create(_email(), "password123")

    conv_id = convdb.save(
        alice["id"], None, "bm25", [{"role": "user", "content": "okapi bm25 saturation"}]
    )

    assert convdb.get(conv_id, alice["id"])["title"] == "okapi bm25 saturation"
    assert convdb.get(conv_id, bob["id"]) is None, "read another user's conversation"
    assert convdb.delete(conv_id, bob["id"]) is False, "deleted another user's thread"
    assert [c["id"] for c in convdb.list_for_user(bob["id"])] == []
    # Bob passing Alice's id to save() must not overwrite her thread.
    bobs = convdb.save(bob["id"], conv_id, "vsm", [{"role": "user", "content": "mine"}])
    assert bobs != conv_id, "wrote into another user's conversation"
    assert convdb.get(conv_id, alice["id"])["title"] == "okapi bm25 saturation"
    assert convdb.delete(conv_id, alice["id"]) is True
    print("  cross-account read, delete and overwrite all refused")


if __name__ == "__main__":
    for fn in (
        test_hash_is_salted_and_verifiable,
        test_a_refresh_token_is_not_an_access_token,
        test_register_never_produces_an_admin,
        test_role_gates_on_document_routes,
        test_history_is_not_readable_across_accounts,
    ):
        print(fn.__name__)
        fn()
    print("\nall phase-6 checks passed")
