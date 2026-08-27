"""Which LLMs this API key can reach.

The UI's model picker is populated from the provider rather than a hardcoded
list, so adding model access on the provider side shows up in the dropdown
without a code change.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.engines.semantic import llm

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    available = llm.available_models()
    return {
        "configured": bool(settings.llm_api_key),
        "base_url": settings.llm_base_url,
        "default": settings.llm_model,
        "models": available,
        "note": None
        if available
        else "Could not list models — check LLM_API_KEY and LLM_BASE_URL, or the "
        "provider may not implement /models. You can still set LLM_MODEL manually.",
    }
