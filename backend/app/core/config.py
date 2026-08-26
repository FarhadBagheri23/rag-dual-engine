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
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    # --- storage ---
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "index"

    # --- chunking (RAG slides s32: recursive splitter, respects paragraph bounds) ---
    chunk_size: int = 500  # SPEC 500 / SLIDES 1000
    chunk_overlap: int = 50  # SPEC  50 / SLIDES  150

    # --- retrieval ---
    top_k: int = 10
    champion_r: int = 50  # 8-Scoring s26: r docs of highest tf per term

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

    # --- embeddings (RAG slides s33) ---
    embed_model: str = "all-MiniLM-L6-v2"  # SPEC MiniLM / SLIDES nomic-embed-text


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
