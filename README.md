# Dual-Engine Document Search & RAG

A document search system that runs the same corpus through two parallel engines
and keeps them in sync: a **classical lexical index** built from scratch
(TF-IDF / BM25 over a hand-rolled inverted index) and a **dense semantic index**
(embeddings in a vector DB) feeding a **RAG** pipeline that answers in natural
language with citations.

Modern Information Retrieval — Sharif University of Technology, International Campus.

## Architecture

```
        ┌──────────────────────── React (Vite + Tailwind) ─────────────────────────┐
        │  Sidebar: per-user history, tagged VSM / BM25 / RAG                       │
        │  Admin (admin only): upload · corpus table · delete                       │
        │  Search: query · engine · PRF toggle · results | RAG chat                 │
        └──────────────────────────────────┬───────────────────────────────────────┘
                                           │  /api   Authorization: Bearer <access>
        ┌──────────────────────────────────▼───────────────────────────────────────┐
        │                              FastAPI                                      │
        │   get_current_user ─── guards search, listing, history                    │
        │   get_current_admin ── guards upload and delete                           │
        │   api/auth.py   api/documents.py   api/search.py   api/conversations.py   │
        └───────┬───────────────────────┬───────────────────────────┬──────────────┘
                │ write                 │ read                      │ read/write
        ┌───────▼────────┐   ┌──────────▼──────────┐                │
        │   corpus.py    │   │  lexical/ semantic/ │                │
        │  SINGLE WRITER │   │   (pure readers)    │                │
        └───┬────────┬───┘   └─────────────────────┘                │
 add/delete │        │ add/delete                                   │
    ┌───────▼──┐  ┌──▼────────────┐          ┌────────────────────▼─────────────┐
    │ Inverted │  │  Vector DB    │          │  SQLite  meta.db                  │
    │  index   │  │  (Chroma)     │◀─derived─│  documents · chunks               │
    └──────────┘  └───────────────┘  from    │  users · conversations · messages │
         ▲                                   └───────────────────────────────────┘
         └────────────────── derived from ───────────────┘
```

**One database, two derived indexes.** Accounts and history live in the same
`meta.db` as the corpus, not in a second database file. The inverted index and
Chroma are *caches* — both are rebuilt from SQLite at startup whenever they
drift. A user row is not rebuildable from anything, so it belongs in the source
of truth: one file to back up, and foreign keys that actually resolve.

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
| 5 | React admin + search UI | ✅ |
| 6 | JWT accounts, client/admin roles, per-user history | ✅ |
| 7 | Bonus: URL scraping, visualization, reranking | ☐ |
| 8 | Evaluation, Docker, docs | ☐ |

## Accounts and roles

Two roles, and the difference between them is exactly one thing: who may change
the corpus.

| | search & chat | read the corpus listing | upload / delete documents |
|---|---|---|---|
| **client** | ✅ | ✅ | ❌ |
| **admin** | ✅ | ✅ | ✅ |

`POST /api/auth/register` always creates a **client** — the role is not a field
it reads, so no request body can talk it into minting an admin. The single admin
is seeded at startup from `ADMIN_EMAIL` / `ADMIN_PASSWORD`, which makes the
environment the only place an admin can come from.

Passwords are **bcrypt** hashes. `gensalt()` draws 16 fresh random bytes per
password and `hashpw` stores them inside the hash string, so two users with the
same password still get different hashes and one leaked table cannot be attacked
with a single precomputed table. There is no separate salt column because there
does not need to be — the salt travels in the hash.

Authentication is a **30-minute access token** plus a **7-day refresh token**,
both JWTs signed with `JWT_SECRET`. The `kind` claim is inside the payload, not
merely implied by the lifetime: without it the two tokens are indistinguishable
to the verifier and the 30-minute limit becomes seven days. The frontend
refreshes on a 401 and replays the request, so expiry is invisible until the
refresh token itself dies.

There is deliberately **no refresh-token table**. Storing them would buy
revocation, at the cost of a lookup on every refresh and a logout that has to
reach the server. Rotating `JWT_SECRET` is the global sign-out.

`JWT_SECRET` has **no default value** in `config.py`, so a missing secret is a
startup crash rather than an app quietly signing tokens with a value that is in
the repository. Generate one with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## History

Every search and every chat is saved against the account that ran it and listed
in the sidebar, tagged with the engine that answered it. Clicking a row restores
it; the two engine families restore differently, on purpose:

- **VSM and BM25** store only the query and how it was run (`mode`, `prf`). A
  lexical search replays in under a millisecond, and stored snippets would go
  stale the moment an admin deletes the document they came from.
- **RAG** stores the answer in full, with its citations and retrieved passages.
  Regenerating one costs an LLM call and would not come back the same anyway, so
  a replay has to be an actual replay.

Ownership is a `WHERE user_id = ?` on every query rather than a check in the
route — there is no function in `database/conversations.py` that reads a thread
without being told whose it is. Missing and not-yours both return 404, because
"that exists but is not yours" leaks that it exists.

## Running it

**There is no database server to start.** SQLite is a file and ChromaDB runs
embedded in the backend process, so the whole system is two commands. Both
stores live under `backend/data/index/` and are created on first run.

### One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # then fill in LLM_API_KEY, JWT_SECRET, ADMIN_PASSWORD
(cd frontend && npm install)
```

`JWT_SECRET` is required — the backend refuses to start without it:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The first backend start downloads the embedding model (~470 MB) into the
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
python -m tests.test_auth
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
| backend won't start, `jwt_secret Field required` | `JWT_SECRET` missing from `.env` — this is deliberate, not a bug |
| Admin tab missing | you are signed in as a client; the seeded admin is `ADMIN_EMAIL` |
| every request 401s after ~30 min idle | refresh token expired too (7 days) — sign in again |
| `403 This action requires an admin account` | a client tried to upload or delete; the gate is working |

## API

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | open | status + which LLM is configured |
| `POST` | `/api/auth/register` | open | create a **client** account, returns both tokens |
| `POST` | `/api/auth/login` | open | exchange credentials for both tokens |
| `POST` | `/api/auth/refresh` | open | 7-day refresh token → a fresh 30-minute access token |
| `GET` | `/api/auth/me` | client | who the current token belongs to |
| `GET` | `/api/documents` | client | corpus listing |
| `POST` | `/api/documents` | **admin** | upload a `.pdf` or `.docx` |
| `DELETE` | `/api/documents/{id}` | **admin** | remove from **every** index, reports what each removed |
| `POST` | `/api/search` | client | `{query, engine, mode, prf, model, k}` → ranked hits |
| `POST` | `/api/search/stream` | client | RAG as NDJSON events |
| `GET` | `/api/models` | client | LLM ids this key can reach, for the model picker |
| `GET` | `/api/conversations` | client | this user's history, for the sidebar |
| `GET` | `/api/conversations/{id}` | client | one thread with its messages |
| `POST` | `/api/conversations` | client | create or replace a thread |
| `DELETE` | `/api/conversations/{id}` | client | delete a thread (messages go by CASCADE) |

"client" means any signed-in account, admins included. Unauthenticated requests
get **401**; a client hitting an admin route gets **403**, not 401 — they are
correctly signed in, so sending them back to the login screen would be a lie.

## Retrieval

### The retrieval unit is the document

Chunks exist because embedding models have a context window (spec §2.1) — they
are not a thing anyone searches for. The spec is consistent about the unit
everywhere it matters: VSM and BM25 "display a ranked list of relevant
**documents**" (§3.3.2), use "TF-IDF weighting for queries and **documents**"
(§3.2.1), eliminate by "only scoring **documents** that contain high-IDF query
terms", and "assume the top *k* **documents** are relevant" for PRF.

So the index keeps both levels, built in one pass:

| level | holds | used by |
|---|---|---|
| chunk | `postings[term] = {chunk_id: tf}`, forward index, per-chunk norm | §2.1 indexing, RAG retrieval, choosing each result's snippet |
| document | tf summed over the document's chunks, doc frequency, doc length, per-document norm | VSM, BM25, Rocchio — everything that produces a score |

Scoring chunks instead is wrong in three compounding ways, and the third is the
one that is easy to miss:

1. **A document appears several times in one ranked list**, its chunks competing
   against each other for result slots.
2. **`df` stops meaning document frequency.** idf over chunks makes a term's
   apparent rarity depend on how the corpus happened to be split.
3. **BM25's `b` is left with nothing to normalize.** `b` exists to stop long
   documents winning on raw term count — but chunking deliberately makes every
   chunk about one length. On this corpus, chunk lengths vary with a coefficient
   of variation of 0.36 against 0.69 for documents, so normalizing by chunk
   length corrects for variance chunking had already removed.

`tests/test_lexical.py::test_a_document_is_ranked_once_and_scored_whole` is the
regression guard: a three-chunk document and a one-chunk document, asserting two
ranked rows and a summed tf.

### Weighting

Scoring is **lnc.ltc**, a very standard weighting scheme — documents get log
tf, no idf, cosine normalization; queries get log tf, idf, cosine
normalization. `tests/test_lexical.py` reproduces the worked example. Top-K selection uses a
binary min-heap, O(N log K) rather than sorting all N.

One detail worth stating outright, because it differs from a naive reading:
**documents carry no idf** — that is the `n` in `lnc`. idf measures a term's
rarity across the collection, which is a property of the term and not of any one
document, so weighting both sides by it squares the effect. Putting idf on the
query side only is what lnc.ltc means.

Three retrieval modes, two of them inexact top-K:

| `mode` | Contender set A | Safe? |
|---|---|---|
| `exact` | every document containing a query term | yes |
| `champion` | champion lists — the `r` highest-tf documents per term | no |
| `elimination` | all postings, but only for high-idf query terms | no |

`scored` in the response is how many documents were actually scored, so the
saving from an inexact mode is visible per query rather than merely claimed.
With PRF it is the total over **two** retrievals and can exceed the collection
size — a document in both candidate sets is scored twice — so the response also
carries `passes`, the per-retrieval split, and the UI renders "6 + 7 of 9"
rather than an arithmetically impossible "13 of 9".
All three modes work with both lexical engines, with two honest caveats:

- **`elimination` scores are not comparable to `exact` scores.** The method is to
  "only accumulate scores from catcher and rye", so the dropped terms'
  contributions are omitted rather than renormalized away — the score is a
  genuine partial cosine, and it is systematically lower. The ranking is the
  point; the magnitude is not comparable across modes, and the UI says so.
- **`champion` is a looser approximation under BM25 than under VSM.** Champion lists
  scopes champion lists to tf-idf, where score rises monotonically with tf.
  BM25 also divides by document length, so a high-tf long document can rank
  below a low-tf short one the tf-ordered list omitted. Both modes are non-safe
  by construction; this combination is simply less tight.

**BM25** adds probabilistic weighting with
tunable `k1` (term frequency saturation), `b` (length normalization) and `k3`
(query-side saturation). `b` and `avgdl` are measured over documents, for the
reason given above. The two models disagree in an instructive way: cosine
normalization punishes a long document harder than BM25's `b=0.75`, so the
same query can rank a different document first under each — which is why the
spec asks for both.

**Pseudo-relevance feedback** runs Rocchio on the assumption that
the top **documents** of an initial retrieval are relevant, as §3.2.1 words it.
Building the centroid from chunks instead would let one long document
contribute several times over, weighting the expansion by how the corpus
happened to be split:

    q_m = α·q_0 + β·centroid(top n) − γ·centroid(next m)

with α=1.0, β=0.75, γ=0.15, negative weights clipped, and the expansion capped
at the heaviest 20 terms because long queries are expensive. Original
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

RAG is the one route that ranks **chunks**, deliberately: the passage is what
goes into the prompt, and a citation has to point at the sentence a claim came
from rather than at a whole PDF. So one document legitimately supplies several
`[Doc N]` entries — the opposite of what the lexical list must do.

Retrieved text is treated as untrusted input: the passages are
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

**Embeddings run locally on CPU** and need no key, so semantic retrieval works
offline and for free — only the generated answer requires a provider. The model
is `paraphrase-multilingual-MiniLM-L12-v2`, not the spec's `all-MiniLM-L6-v2`:
the latter is English-only, and a Persian query scored 0.07 against the passage
that answers it where the English phrasing scored 0.46 — noise, not retrieval.
The multilingual sibling aligns 50+ languages in one space, so a Persian
question retrieves an English passage. It costs ~470 MB against ~90 MB.

## Layout

```
backend/app/
  main.py          app factory: CORS, routers, admin seeding
  core/            cross-cutting: config.py (every tunable), exceptions.py,
                   security.py (bcrypt hashing + JWT mint/verify)
  database/        SQLite: session.py (connection + schema), and one module
                   per table group — documents.py, users.py, conversations.py
  services/        business logic: ingest.py (pdf/docx -> chunks),
                   corpus.py (THE single writer to both indexes)
  engines/         the two retrieval engines + registry.py dispatch
    lexical/       text.py, index.py, heap.py, vsm.py, bm25.py
    semantic/      embedder.py, vectordb.py, llm.py, rag.py
  api/             documents.py (CRUD), search.py, conversations.py,
                   auth.py (routes + get_current_user / get_current_admin)
  schemas.py       pydantic request/response models
frontend/src/
  api.js           one authed fetch: bearer header + refresh-and-replay
  App.jsx          auth gate, sidebar layout, conversation state
  pages/           Login.jsx, Search.jsx, Admin.jsx
  components/      Sidebar.jsx (history), Chat.jsx, Results.jsx, …
```

The dependency direction is one-way: `api` → `services` → (`database`, `engines`)
→ `core`. Nothing lower ever imports from something higher, which is what keeps
`corpus.py` the only path that can mutate an index.
