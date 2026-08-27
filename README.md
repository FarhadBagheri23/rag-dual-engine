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
| 4 | Embeddings, Chroma, RAG with citations | ✅ |
| 5 | React admin + search UI | ☐ |
| 6 | Bonus: URL scraping, visualization, reranking | ☐ |
| 7 | Evaluation, Docker, docs | ☐ |

## Running it

**There is no database server to start.** SQLite is a file and ChromaDB runs
embedded in the backend process, so the whole system is two commands. Both
stores live under `backend/data/index/` and are created on first run.

### One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # then put your key in LLM_API_KEY
(cd frontend && npm install)
```

The first backend start downloads the embedding model (~90 MB) into the
HuggingFace cache. Everything after that is offline except LLM generation.

### Every time — two terminals

Terminal 1, backend on `http://localhost:8000`:

```bash
cd backend && source ../venv/bin/activate && uvicorn app.main:app --reload
```

Terminal 2, frontend on `http://localhost:5173`:

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173**. The frontend proxies `/api` to port 8000, so
the backend must be running or the status dot goes red. Interactive API docs
are at **http://localhost:8000/docs**.

### Checks

```bash
cd backend && source ../venv/bin/activate
python -m tests.test_ingest
python -m tests.test_lexical
python -m tests.test_bm25_prf
python -m tests.test_rag_sync
```

### Starting over

Both indexes are derived from SQLite, and SQLite is derived from the uploaded
files — so deleting state is safe and recoverable at each level:

```bash
rm backend/data/index/inverted.pkl     # rebuilt from SQLite at next startup
rm -rf backend/data/index/             # full reset: re-upload the documents
```

### If something is wrong

| Symptom | Cause |
|---|---|
| status dot red | backend not running, or not on port 8000 |
| `GET /api/models` returns `[]` | `LLM_API_KEY` empty, or wrong `LLM_BASE_URL` |
| RAG answers `null` with a `note` | no key set — retrieval still works, generation does not |
| `Address already in use` | an old `uvicorn` is alive: `pkill -f "uvicorn app"` |
| search returns nothing after a restart | index missing; it rebuilds at startup, check the log |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | status + which LLM is configured |
| `GET` | `/api/documents` | corpus listing |
| `POST` | `/api/documents` | upload a `.pdf` or `.docx` |
| `DELETE` | `/api/documents/{id}` | remove from **every** index, reports what each removed |
| `POST` | `/api/search` | `{query, engine, mode, prf, model, k}` → ranked hits |
| `GET` | `/api/models` | LLM ids this key can reach, for the model picker |

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

**RAG** embeds the query with the same model used at index time, takes the
nearest chunks from Chroma by cosine similarity, numbers them `[Doc 1] …
[Doc N]` in the prompt, and asks the model to answer only from those passages
and cite each claim. Citation markers are parsed back out of the answer and
resolved to the chunks that produced them, so every claim is traceable.

Retrieved text is treated as untrusted input (RAG slides s44): the passages are
fenced in the prompt and the instructions state they are reference material,
never commands. A document in the corpus cannot instruct the model.

## Synchronization

`services/corpus.py` is the only module that writes. `add()` and `remove()`
each touch all three stores in one place:

| Store | Holds | Keyed by |
|---|---|---|
| SQLite | document metadata and chunk **text** — the source of truth | `chunk_id` |
| Inverted index | postings, doc lengths, norms | `chunk_id` |
| Chroma | dense vectors | `chunk_id` |

Because all three key on the same `chunk_id`, deletion is a lookup rather than
a search. `DELETE /api/documents/{id}` returns what each store removed:

```json
{"chunks_removed": 1, "postings_removed": 174, "vectors_removed": 1}
```

Both derived stores are **reconciled against SQLite at startup**, so drift is
repairable rather than fatal. Delete `inverted.pkl` or the whole `chroma/`
directory and the next start rebuilds it. This is not hypothetical: the sample
corpus was indexed before the vector store existed, and the reconcile is what
made those five chunks reachable by semantic search without re-uploading them.

The Vite dev server proxies `/api` to `http://localhost:8000`, so no API base
URL needs configuring.

### LLM provider

Any OpenAI-compatible endpoint works — set three variables in `.env`:

| Provider | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| AvalAI | `https://api.avalai.ir/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
| Ollama | `http://localhost:11434/v1` | `llama3.2` |

`GET /api/models` asks the provider which models the key can actually reach, so
the UI's model picker reflects the account rather than a hardcoded list. If a
key gains access to more models, they appear without a code change.

**Embeddings run locally on CPU** (`all-MiniLM-L6-v2`) and need no key, so
semantic retrieval works offline and for free — only the generated answer
requires a provider. It also keeps the vectors deterministic, which is what
lets `eval/` compare runs.

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
