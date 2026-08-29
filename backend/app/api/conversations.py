"""Per-user search and chat history — what the sidebar reads.

Every route takes the current user from the token and passes their id down to
the query. The router-level dependency makes that non-optional: there is no
unauthenticated way in, so a forgotten `Depends` on one route cannot expose
another user's threads.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import get_current_user
from app.database import conversations as db
from app.schemas import ConversationDetail, ConversationOut, ConversationSave

router = APIRouter(
    prefix="/api/conversations",
    tags=["conversations"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=list[ConversationOut])
def list_conversations(user: dict = Depends(get_current_user)):
    return db.list_for_user(user["id"])


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    conv = db.get(conv_id, user["id"])
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation")
    return conv


@router.post("", response_model=ConversationOut)
def save_conversation(body: ConversationSave, user: dict = Depends(get_current_user)):
    """Create or update one thread. The client sends the whole transcript and
    gets the row back, so it learns the id of a thread it just started."""
    if not body.messages:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "A conversation needs a message"
        )
    conv_id = db.save(
        user["id"], body.id, body.engine, [m.model_dump() for m in body.messages]
    )
    return db.get(conv_id, user["id"])


@router.delete("/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    if not db.delete(conv_id, user["id"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such conversation")
