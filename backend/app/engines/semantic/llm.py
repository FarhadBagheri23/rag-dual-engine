"""One client for any OpenAI-compatible provider — RAG slides s30.

AvalAI, Groq, Gemini's compatibility layer, Ollama and OpenAI itself all speak
the same wire format, so `LLM_BASE_URL` is the only thing that changes. That is
also why the key is named LLM_API_KEY rather than after any one vendor.

`available_models()` asks the provider what the key can actually reach, so the
UI's model picker reflects the account instead of a hardcoded list.
"""

from functools import lru_cache

from openai import APIError, APIStatusError, OpenAI

from app.core.config import settings
from app.core.exceptions import ProviderError


@lru_cache(maxsize=1)
def client() -> OpenAI:
    return OpenAI(
        api_key=settings.llm_api_key or "not-set",
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
    )


def available_models() -> list[str]:
    """Model ids this key can use. Empty list if the provider cannot be asked."""
    try:
        return sorted(m.id for m in client().models.list().data)
    except Exception:
        # No key, wrong base URL, or a provider that does not implement
        # /models. The caller decides what to show; this is not fatal.
        return []


def complete(prompt: str, model: str | None = None) -> str:
    try:
        response = client().chat.completions.create(
            model=model or settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # grounded answers should not be creative
        )
    except APIStatusError as exc:
        # The provider's own message is the useful part — a restricted key or
        # an unknown model says exactly what to do. Pass it through.
        detail = exc.message
        try:
            detail = exc.response.json()["error"]["message"]
        except Exception:
            pass
        raise ProviderError(exc.status_code, detail) from None
    except APIError as exc:  # network, timeout, malformed response
        raise ProviderError(None, str(exc)) from None

    return response.choices[0].message.content or ""
