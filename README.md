# Dual-Engine Document Search & RAG

A document search system that runs the same corpus through two parallel engines
and keeps them in sync: a **classical lexical index** built from scratch
(TF-IDF / BM25 over a hand-rolled inverted index) and a **dense semantic index**
(embeddings in a vector DB) feeding a **RAG** pipeline that answers in natural
language with citations.

Modern Information Retrieval — Sharif University of Technology, International Campus.

## Architecture

```
                 ┌─────────────────────── React (Vite + Tailwind) ──────────────────────┐
                 │   Admin: upload · corpus table · delete    Search: query · engine ·   │
                 │                                            PRF toggle · results       │
                 └───────────────────────────────┬──────────────────────────────────────┘
                                                 │  /api
                 ┌───────────────────────────────▼──────────────────────────────────────┐
                 │                          FastAPI                                      │
                 │   api/documents.py            api/search.py                           │
                 └───────┬───────────────────────────────────┬──────────────────────────┘
                         │ write                             │ read
                 ┌───────▼────────┐              ┌───────────▼───────────┐
                 │   corpus.py    │              │   lexical/  semantic/ │
                 │  SINGLE WRITER │              │   (pure readers)      │
                 └───┬────────┬───┘              └───────────────────────┘
          add/delete │        │ add/delete
             ┌───────▼──┐  ┌──▼────────────┐
             │ Inverted │  │  Vector DB    │
             │  index   │  │  (Chroma)     │
             └──────────┘  └───────────────┘
```

**The one architectural rule:** `corpus.py` is the only module that writes. Both
indexes are mutated from a single function, so they cannot drift apart — that is
what makes deletion provably synchronized rather than hopefully synchronized.
The retrieval engines never write; they only read what `corpus.py` built.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, config, health check | ✅ |
| 1 | Parsing, chunking, corpus CRUD | ✅ |
| 2 | Inverted index, VSM (lnc.ltc), inexact top-K | ✅ |
| 3 | BM25, Rocchio pseudo-relevance feedback | ✅ |
| 4 | Embeddings, Chroma, RAG with citations | ☐ |
| 5 | React admin + search UI | ☐ |
| 6 | Bonus: URL scraping, visualization, reranking | ☐ |
| 7 | Evaluation, Docker, docs | ☐ |

## Setup

```bash
cp .env.example .env      # then fill in LLM_API_KEY
```

Backend:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

Checks:

```bash
cd backend && python -m tests.test_ingest && python -m tests.test_lexical && python -m tests.test_bm25_prf
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | status + which LLM is configured |
| `GET` | `/api/documents` | corpus listing |
| `POST` | `/api/documents` | upload a `.pdf` or `.docx` |
| `DELETE` | `/api/documents/{id}` | remove from **every** index, reports what each removed |
| `POST` | `/api/search` | `{query, engine, mode, k}` → ranked hits with snippets |

## Retrieval

The inverted index is built from scratch: `postings[term] = {chunk_id: tf}`,
plus a forward index for O(terms) deletion and a precomputed L2 norm per chunk.

Scoring is **lnc.ltc** (slide 7-Scoring s42) — documents get log tf, no idf,
cosine normalization; queries get log tf, idf, cosine normalization. Top-K
selection uses a binary min-heap, O(N log K) rather than sorting all N
(slide 8-Scoring s13–14).

Three retrieval modes, two of them inexact top-K (slide 8-Scoring s19):

| `mode` | Contender set A | Safe? |
|---|---|---|
| `exact` | every chunk containing a query term | yes |
| `champion` | champion lists — the `r` highest-tf chunks per term (s26) | no |
| `elimination` | all postings, but only for high-idf query terms (s24) | no |

`scored` in the response is how many chunks were actually visited, so the
saving from an inexact mode is visible per query rather than merely claimed.
All three modes work with both lexical engines.

**BM25** (slide 11-Probabilistic s31–32) adds probabilistic weighting with
tunable `k1` (term frequency saturation), `b` (length normalization) and `k3`
(query-side saturation). The two models disagree in an instructive way: cosine
normalization punishes a long document harder than BM25's `b=0.75`, so the
same query can rank a different chunk first under each — which is why the
spec asks for both.

**Pseudo-relevance feedback** (slide 10) runs Rocchio on the assumption that
the top hits of an initial retrieval are relevant:

    q_m = α·q_0 + β·centroid(top n) − γ·centroid(next m)

with α=1.0, β=0.75, γ=0.15, negative weights clipped, and the expansion capped
at the heaviest 20 terms because long queries are expensive (s20). Original
query terms are never capped away. The response returns the added terms in
`expansion`, so the effect is inspectable rather than invisible. PRF is
available on VSM only — Rocchio is defined on the vector space model — and the
route rejects the combination with BM25 rather than silently ignoring it.

The index is **derived** from SQLite: delete `data/index/inverted.pkl` and it
is rebuilt at startup with identical rankings.

The Vite dev server proxies `/api` to `http://localhost:8000`, so no API base
URL needs configuring.

### LLM provider

Any OpenAI-compatible endpoint works — set three variables in `.env`:

| Provider | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| Ollama | `http://localhost:11434/v1` | `llama3.2` |

## Layout

```
backend/app/
  main.py          app factory: CORS, routers
  core/            cross-cutting: config.py (every tunable), exceptions.py
  database/        SQLite metadata: session.py (connection + schema),
                   documents.py (all the SQL, in one place)
  services/        business logic: ingest.py (pdf/docx/url -> chunks),
                   corpus.py (THE single writer to both indexes)
  engines/         the two retrieval engines + registry.py dispatch
    lexical/       text.py, index.py, heap.py, vsm.py, bm25.py
    semantic/      embedder.py, vectordb.py, llm.py, rag.py
  api/             documents.py (CRUD), search.py — routing only, no logic
  schemas.py       pydantic request/response models
frontend/src/      pages/ + components/
eval/              retrieval metrics notebook
```

The dependency direction is one-way: `api` → `services` → (`database`, `engines`)
→ `core`. Nothing lower ever imports from something higher, which is what keeps
`corpus.py` the only path that can mutate an index.
