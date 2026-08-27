"""Local sentence-transformers embeddings — the spec's recommended stack.

Embedding runs on CPU with no API key, so semantic *retrieval* works offline
and for free; only the generation step in rag.py needs a provider. It also
keeps the vectors deterministic, which matters for eval/: the same corpus
always produces the same index.

The model is loaded lazily. Importing this module must stay cheap, because
main.py imports the whole engine tree at startup and a ~90 MB model download
should not block a health check.
"""

from functools import lru_cache

from app.core.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # heavy, import late

    return SentenceTransformer(settings.embed_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(text, normalize_embeddings=True).tolist()


def dimensions() -> int:
    return _model().get_sentence_embedding_dimension()
