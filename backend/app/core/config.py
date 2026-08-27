"""Every tunable in one place.

Values that differ between the project spec and the course slides are marked
SPEC / SLIDES — the default follows the spec (it is the graded document) and
eval/evaluation.ipynb sweeps the alternative.
"""

from pathlib import Path

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
    rag_top_k: int = 4  # chunks placed in the prompt; keep small (s43)
    # RAG slides s11: dense retrieval always returns *something*, so an
    # out-of-corpus question still gets k neighbours — "set a max similarity
    # threshold". Measured on this corpus: in-scope questions score 0.31-0.71,
    # out-of-scope -0.07-0.15. 0.22 sits in the gap.
    rag_min_score: float = 0.22

    # --- storage ---
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "index"

    # --- ingestion ---
    max_upload_bytes: int = 50_000_000

    # --- chunking (RAG slides s32: recursive splitter, respects paragraph bounds) ---
    # Measured in WORDS, because the spec says "500 words with a 50-word overlap".
    # The slides' lab used 1000/150 *characters* (~160/24 words); eval/ sweeps both.
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- retrieval ---
    top_k: int = 10
    champion_r: int = 50  # 8-Scoring s26: r docs of highest tf per term
    # 8-Scoring s24: keep query terms whose idf is >= this fraction of the
    # query's highest idf. 0 keeps everything, 1 keeps only the rarest term.
    elimination_ratio: float = 0.3

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
