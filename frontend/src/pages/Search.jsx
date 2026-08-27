import { useEffect, useRef, useState } from "react";
import * as api from "../api";
import SearchControls from "../components/SearchControls";
import { RagAnswer, ResultCard } from "../components/Results";
import { Alert, Search as SearchIcon, Spinner } from "../components/icons";

const EXAMPLES = [
  "how does term frequency saturate",
  "what stops a language model from making things up",
  "which model should i use for ranking",
];

export default function SearchView({ corpusSize, models }) {
  const [query, setQuery] = useState("");
  const [opts, setOpts] = useState({
    engine: "vsm",
    mode: "champion",
    prf: false,
    model: "",
  });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => inputRef.current?.focus(), []);

  async function run(q = query) {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.search({
          query: q,
          engine: opts.engine,
          mode: opts.mode,
          prf: opts.prf,
          model: opts.engine === "rag" && opts.model ? opts.model : null,
        })
      );
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
        className="mb-5"
      >
        <label htmlFor="q" className="sr-only">
          Search the corpus
        </label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="pointer-events-none absolute left-3.5 top-1/2 h-5 w-5 -translate-y-1/2 text-dim" />
            <input
              id="q"
              ref={inputRef}
              dir="auto"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask the corpus…"
              className="w-full rounded-lg border border-line bg-card py-3 pl-11 pr-4 text-lg placeholder:text-dim"
            />
          </div>
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="cursor-pointer rounded-lg bg-accent px-6 font-medium text-accent-ink transition-colors duration-150 hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? <Spinner className="h-5 w-5" /> : "Search"}
          </button>
        </div>
      </form>

      <div className="mb-6">
        <SearchControls opts={opts} set={setOpts} models={models} />
      </div>

      {error && (
        <div className="card mb-5 flex items-start gap-2.5 border-danger/50 p-4 text-sm">
          <Alert className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
          <span>{error}</span>
        </div>
      )}

      <div aria-busy={busy} aria-live="polite">
        {busy && (
          <ul className="space-y-3">
            {[0, 1, 2].map((i) => (
              <li key={i} className="card animate-pulse p-4">
                <div className="h-4 w-2/5 rounded bg-muted" />
                <div className="mt-3 h-3 w-full rounded bg-muted" />
                <div className="mt-2 h-3 w-4/5 rounded bg-muted" />
              </li>
            ))}
          </ul>
        )}

        {!busy && result && (
          <>
            <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-xs text-dim">
              <span>
                <span className="text-fg">{result.hits.length}</span> hits
              </span>
              <span>
                scored <span className="text-fg">{result.scored}</span> of{" "}
                {corpusSize} chunks
              </span>
              <span>
                <span className="text-fg">{result.took_ms.toFixed(1)}</span> ms
              </span>
              {result.prf && result.expansion.length > 0 && (
                <span>
                  Rocchio added{" "}
                  <span className="text-accent">{result.expansion.length}</span>{" "}
                  terms
                </span>
              )}
            </div>

            {result.prf && result.expansion.length > 0 && (
              <div className="card mb-5 p-4">
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-dim">
                  Query expansion
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {result.expansion.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-accent/15 px-1.5 py-0.5 font-mono text-xs text-accent"
                    >
                      +{t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {result.engine === "rag" && <RagAnswer result={result} />}

            {result.engine !== "rag" &&
              (result.hits.length > 0 ? (
                <ul className="space-y-3">
                  {result.hits.map((h, i) => (
                    <ResultCard key={h.chunk_id} hit={h} rank={i + 1} />
                  ))}
                </ul>
              ) : (
                <div className="card p-8 text-center">
                  <p className="font-medium">No matches for “{result.query}”</p>
                  <p className="mt-1.5 text-sm text-dim">
                    Every query term is absent from the corpus. Try broader
                    wording, or switch to RAG — it matches meaning rather than
                    exact terms.
                  </p>
                </div>
              ))}
          </>
        )}

        {!busy && !result && !error && (
          <div className="card p-8 text-center">
            <SearchIcon className="mx-auto h-8 w-8 text-dim" />
            <p className="mt-3 font-medium">
              {corpusSize === 0
                ? "The corpus is empty"
                : `Search ${corpusSize} indexed chunks`}
            </p>
            {corpusSize === 0 ? (
              <p className="mt-1.5 text-sm text-dim">
                Upload a PDF or Word document from the Admin tab to begin.
              </p>
            ) : (
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((e) => (
                  <button
                    key={e}
                    onClick={() => {
                      setQuery(e);
                      run(e);
                    }}
                    className="cursor-pointer rounded-md border border-line px-3 py-1.5 text-sm text-dim transition-colors duration-150 hover:border-edge hover:text-fg"
                  >
                    {e}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
