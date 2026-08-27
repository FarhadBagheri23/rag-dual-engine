"""Domain errors. FastAPI turns HTTPException into a JSON response for free,
so these are thin subclasses rather than a parallel error hierarchy.
"""

from fastapi import HTTPException, status


class UnsupportedFileType(HTTPException):
    def __init__(self, suffix: str, supported: set[str]):
        super().__init__(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Cannot parse '{suffix}'. Supported: {', '.join(sorted(supported))}",
        )


class FileTooLarge(HTTPException):
    def __init__(self, size: int, limit: int):
        super().__init__(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is {size / 1e6:.1f} MB, limit is {limit / 1e6:.0f} MB",
        )


class DocumentNotFound(HTTPException):
    def __init__(self, doc_id: str):
        super().__init__(status.HTTP_404_NOT_FOUND, f"No document with id '{doc_id}'")


class ProviderError(HTTPException):
    """An LLM provider rejected or failed the request.

    Providers return genuinely useful messages — a restricted key, an unknown
    model, an exhausted quota — and swallowing them into a bare 500 leaves the
    user with nothing to act on. 4xx passes through as 400 because the caller
    can fix it; anything else is 502, because the fault is upstream.
    """

    def __init__(self, status: int | None, message: str):
        super().__init__(
            400 if status and 400 <= status < 500 else 502,
            f"LLM provider error: {message}",
        )


class EmptyDocument(HTTPException):
    def __init__(self, filename: str):
        super().__init__(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"No extractable text in '{filename}' — is it a scanned image?",
        )
