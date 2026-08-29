import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import Chat from "../components/Chat";
import Compare from "../components/Compare";
import Heatmap from "../components/Heatmap";
import SearchControls from "../components/SearchControls";
import { ResultCard } from "../components/Results";
import { Alert, Search as SearchIcon, Spinner } from "../components/icons";

const EXAMPLES = [
  "how does term frequency saturate",
  "what stops a language model from making things up",
  "which model should i use for ranking",
];

export default function SearchView({
  corpusSize, // chunks — what RAG retrieves over
  docCount, // documents — what VSM and BM25 rank
  models,
  conversation,
  onSaved,
}) {
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
  // Opt-in: comparing costs a second retrieval, and most searches are just
  // searches. Closed again on every new query, so it never shows one query's
  // comparison under another query's results.
  const [comparing, setComparing] = useState(false);
  const inputRef = useRef(null);

  const rag = opts.engine === "rag";
  useEffect(() => {
    if (!rag) inputRef.current?.focus();
  }, [rag]);

  /** Run a query. `save` is false when replaying history, so re-opening a
   *  thread does not deposit a duplicate of itself in the sidebar. */
  const execute = useCallback(
    async (q, o, save) => {
      if (!q.trim()) return;
      setBusy(true);
      setError(null);
      setComparing(false);
      try {
        const res = await api.search({
          query: q,
          engine: o.engine,
          mode: o.mode,
          prf: o.prf,
        });
        setResult(res);
        if (save) {
          // Only the query and how it was run are stored, not the hits: a
          // lexical search is sub-millisecond to replay, and stored snippets
          // would go stale the moment an admin deletes the document. RAG is
          // the opposite case and does store its answer.
          const conv = await api
            .saveConversation({
              engine: o.engine,
              messages: [
                {
                  role: "user",
                  content: q,
                  meta: { mode: o.mode, prf: o.prf },
                },
              ],
            })
            .catch(() => null);
          if (conv) onSaved(conv);
        }
      } catch (e) {
        setError(e.message);
        setResult(null);
      } finally {
        setBusy(false);
      }
    },
    [onSaved]
  );

  // Replay a thread picked from the sidebar. RAG restores from its stored
  // transcript inside Chat; the lexical engines just re-run the query.
  useEffect(() => {
    if (!conversation) {
      setQuery("");
      setResult(null);
      return;
    }
    const first = conversation.messages[0];
    const restored = { engine: conversation.engine, ...(first?.meta ?? {}) };
    setOpts((o) => ({ ...o, ...restored }));
    if (conversation.engine !== "rag" && first) {
      setQuery(first.content);
      execute(first.content, { ...opts, ...restored }, false);
    }
    // opts is deliberately not a dependency: this runs when a *thread* is
    // opened, not every time a control moves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversation, execute]);

  const run = (q = query) => execute(q, opts, true);

  return (
    <div>
      <div className="mb-6">
        <SearchControls opts={opts} set={setOpts} models={models} />
      </div>

      {/* RAG is a conversation, not a result list: the question is natural
          language, the reply is prose, and it takes seconds rather than
          milliseconds. It gets its own surface. The lexical engines keep the
          ranked list, which is what they actually produce. */}
      {rag ? (
        <Chat
          // Remount on a model switch or when a different thread is opened, so
          // the transcript state starts from the restored messages rather than
          // being merged into whatever was on screen.
          key={`${opts.model}:${conversation?.id ?? "new"}`}
          model={opts.model || null}
          corpusSize={corpusSize}
          conversation={conversation?.engine === "rag" ? conversation : null}
          onSaved={onSaved}
        />
      ) : (
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
                  placeholder="Search the corpus…"
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
                  {/* Documents, not chunks — the candidate set the inexact
                      modes prune is a set of documents, because that is what
                      gets scored.

                      Rocchio retrieves twice, so its total can exceed the
                      corpus size: a document in both candidate sets is scored
                      twice. Shown per pass ("6 + 7 of 9") rather than summed,
                      because "13 of 9 documents" is not a thing. */}
                  <span
                    title={
                      (result.passes?.length ?? 0) > 1
                        ? "Rocchio retrieves twice: once to pick the feedback set, once with the expanded query."
                        : "Documents whose score was actually computed, out of the whole corpus."
                    }
                  >
                    scored{" "}
                    <span className="text-fg">
                      {(result.passes?.length ?? 0) > 1
                        ? result.passes.join(" + ")
                        : result.scored}
                    </span>{" "}
                    of {docCount} documents
                    {(result.passes?.length ?? 0) > 1 && " (two retrievals)"}
                  </span>
                  <span>
                    <span className="text-fg">{result.took_ms.toFixed(1)}</span>{" "}
                    ms
                  </span>
                  {result.prf && result.expansion.length > 0 && (
                    <span>
                      Rocchio added{" "}
                      <span className="text-accent">
                        {result.expansion.length}
                      </span>{" "}
                      terms
                    </span>
                  )}
                  {/* Elimination omits the low-idf terms' contributions rather
                      than renormalizing (slide 8-Scoring s24), so its scores
                      are genuinely lower than exact's for the same query. Say
                      so where the numbers are, or it reads as a bug. */}
                  {result.mode === "elimination" && result.hits.length > 0 && (
                    <span title="Dropped terms contribute nothing, so scores are lower than Exact by construction. Ranking is what this mode approximates, not magnitude.">
                      scores not comparable to Exact
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

                {result.hits.length > 0 ? (
                  <>
                    <ul className="space-y-3">
                      {/* Keyed by document — VSM and BM25 rank documents, so a
                          doc_id appears at most once in this list. */}
                      {result.hits.map((h, i) => (
                        <ResultCard key={h.doc_id} hit={h} rank={i + 1} />
                      ))}
                    </ul>

                    {/* The ranked list says which documents won; the matrix
                        says which term won them. Column i belongs to result i
                        above, and the two are read together. */}
                    <div className="mt-5">
                      <Heatmap
                        engine={result.engine}
                        terms={result.terms}
                        hits={result.hits}
                      />
                    </div>

                    {!comparing && (
                      <button
                        onClick={() => setComparing(true)}
                        className="mt-4 w-full cursor-pointer rounded-lg border border-dashed border-line py-3 text-sm text-dim transition-colors duration-150 hover:border-edge hover:text-fg"
                      >
                        Run this query on{" "}
                        <span className="font-medium">
                          {result.engine === "vsm" ? "BM25" : "VSM"}
                        </span>{" "}
                        too and compare
                      </button>
                    )}

                    {comparing && (
                      <Compare
                        query={result.query}
                        mode={result.mode}
                        docCount={docCount}
                        onClose={() => setComparing(false)}
                      />
                    )}
                  </>
                ) : (
                  <div className="card p-8 text-center">
                    <p className="font-medium">
                      No matches for “{result.query}”
                    </p>
                    <p className="mt-1.5 text-sm text-dim">
                      Every query term is absent from the corpus. Try broader
                      wording, or switch to RAG — it matches meaning rather than
                      exact terms.
                    </p>
                  </div>
                )}
              </>
            )}

            {!busy && !result && !error && (
              <div className="card p-8 text-center">
                <SearchIcon className="mx-auto h-8 w-8 text-dim" />
                <p className="mt-3 font-medium">
                  {docCount === 0
                    ? "The corpus is empty"
                    : `Search ${docCount} indexed documents`}
                </p>
                {docCount === 0 ? (
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
      )}
    </div>
  );
}
