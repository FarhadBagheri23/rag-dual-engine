"""Which LLMs this API key can reach.

The UI's model picker is populated from the provider rather than a hardcoded
list, so adding model access on the provider side shows up in the dropdown
without a code change.
"""

from fastapi import APIRouter

from app.engines.semantic import llm

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    # /v1/models reports the provider's catalogue, not this key's grants — a
    # restricted key still lists models it will 403 on. Rather than probe
    # hundreds of ids, let the attempt fail with the provider's own message,
    # which names the fix. An empty list is not an error: /api/health already
    # reports whether a key is configured, and the picker falls back to
    # LLM_MODEL typed by hand.
    return {"models": llm.available_models()}
