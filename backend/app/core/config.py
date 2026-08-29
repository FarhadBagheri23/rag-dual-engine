"""Every tunable in one place.

Values that differ between the project spec and the course slides are marked
SPEC / SLIDES — the default follows the spec, it being the graded document.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR.parent / ".env", BASE_DIR / ".env"), extra="ignore"
    )

    # --- LLM: any OpenAI-compatible endpoint (Groq, Gemini, Ollama, OpenAI) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.avalai.ir/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 60.0
    # Chunks placed in the prompt. s43 says keep the context small, and 4 was
    # that read taken too literally: on a 16-chunk corpus it cut answer-bearing
    # passages at rank 5. Six is still selective and costs ~4k tokens.
    rag_top_k: int = 6
    # ...of which at most this many may come from any one document, so a single
    # file cannot own the whole context. See rag._diversify.
    #
    # Half of rag_top_k, and both neighbours were measured to be worse. At 2 the
    # cap fires when a document genuinely *is* the answer: "what is rocchio"
    # lost its third Relevance-Feedback passage to an unrelated file. At 6 (no
    # cap) one document takes three of four slots and the passages that answer
    # "why normalise document length" fall off the end.
    rag_max_per_doc: int = 3
    # RAG slides s11: dense retrieval always returns *something*, so an
    # out-of-corpus question still gets k neighbours — "set a max similarity
    # threshold". Measured on this corpus: in-scope questions score 0.31-0.71,
    # out-of-scope -0.07-0.15. 0.22 sits in the gap.
    rag_min_score: float = 0.22

    # --- auth ---
    # No default, deliberately: pydantic refuses to construct Settings without
    # JWT_SECRET, so a missing secret is a startup crash rather than an app
    # silently signing tokens with a value that is in the repo.
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 7
    # The single admin, seeded at startup. Blank disables seeding.
    admin_email: str = ""
    admin_password: str = ""

    # --- storage ---
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "index"

    # --- ingestion ---
    max_upload_bytes: int = 50_000_000

    # --- chunking (RAG slides s32: recursive splitter, respects paragraph bounds) ---
    # Measured in WORDS, because the spec says "500 words with a 50-word overlap".
    # SLIDES: the RAG lab used 1000/150 *characters* (~160/24 words).
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- retrieval ---
    top_k: int = 10
    champion_r: int = 50  # 8-Scoring s26: r docs of highest tf per term
    # 8-Scoring s24: keep query terms whose idf is >= this fraction of the
    # query's highest idf. 0 keeps everything, 1 keeps only the rarest term.
    # Bounded at the boundary rather than guarded at the call site: within
    # [0, 1] the highest-idf term always clears its own floor, so elimination
    # can never empty a query. Above 1 it silently could, so pydantic rejects
    # it at startup instead of returning zero hits at query time.
    elimination_ratio: float = Field(0.3, ge=0.0, le=1.0)

    # --- BM25 (11-Probabilistic s31-32; k3 saturates long queries) ---
    bm25_k1: float = 1.5  # SPEC 1.5 / HW2 1.2
    bm25_b: float = 0.75
    bm25_k3: float = 1.2

    # --- Rocchio PRF (10-Relevance Feedback s12) ---
    prf_alpha: float = 1.0
    prf_beta: float = 0.75
    prf_gamma: float = 0.15
    prf_n_relevant: int = 10  # top-n assumed relevant
    prf_n_nonrelevant: int = 20  # next-m assumed nonrelevant
    prf_expansion_terms: int = 20  # s20: cap the expanded query length

    # --- web search (RAG slides s21: retrieval need not be one corpus) ---
    web_top_k: int = 4  # web passages added to the prompt alongside local ones
    web_timeout: float = 8.0  # a slow search must not hold up the whole answer

    # --- embeddings (RAG slides s33) ---
    # SPEC recommends all-MiniLM-L6-v2, but that model is English-only: a
    # Persian query scored 0.07 against the passage that answers it, where the
    # English phrasing scored 0.46 — noise, not retrieval. The multilingual
    # sibling covers 50+ languages and aligns them in one space, so a Persian
    # question can retrieve an English passage. Costs ~470 MB and 384 dims.
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
