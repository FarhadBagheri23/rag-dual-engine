import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.auth import get_current_admin, get_current_user
from app.core.config import settings
from app.core.exceptions import FileTooLarge, UnsupportedFileType
from app.schemas import DeleteResult, DocumentOut
from app.services import corpus
from app.services.ingest import SUPPORTED

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Reading the corpus and changing it are different privileges, so the guard is
# per-route rather than on the router: a client needs the listing (the search
# page shows "N docs · M chunks" from it), but only an admin may add or remove.
ADMIN_ONLY = [Depends(get_current_admin)]


@router.get("", response_model=list[DocumentOut], dependencies=[Depends(get_current_user)])
def list_documents():
    return corpus.listing()


@router.post("", response_model=DocumentOut, status_code=201, dependencies=ADMIN_ONLY)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
):
    """Upload one document. `title` is optional — blank falls back to the
    document's own metadata title, then to its filename."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFileType(suffix or "(none)", SUPPORTED)

    body = await file.read()
    if len(body) > settings.max_upload_bytes:
        raise FileTooLarge(len(body), settings.max_upload_bytes)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        return corpus.add(tmp_path, file.filename, title)
    finally:
        tmp_path.unlink(missing_ok=True)  # no-op once corpus.add has moved it


@router.delete("/{doc_id}", response_model=DeleteResult, dependencies=ADMIN_ONLY)
def delete_document(doc_id: str):
    return corpus.remove(doc_id)
