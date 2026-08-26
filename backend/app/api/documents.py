import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.core.exceptions import FileTooLarge, UnsupportedFileType
from app.schemas import DeleteResult, DocumentOut
from app.services import corpus
from app.services.ingest import SUPPORTED

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents():
    return corpus.listing()


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(file: UploadFile = File(...)):
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
        return corpus.add(tmp_path, file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)  # no-op once corpus.add has moved it


@router.delete("/{doc_id}", response_model=DeleteResult)
def delete_document(doc_id: str):
    return corpus.remove(doc_id)
