import { Alert, Doc, Sparkle } from "./icons";

/** Wrap every matched term in the snippet. Terms come from the backend's own
 *  tokenizer ([a-z0-9]+), so they are already regex-safe. */
function highlight(text, terms) {
  if (!terms?.length) return text;
  const re = new RegExp(`\\b(${terms.join("|")})\\b`, "gi");
  return text.split(re).map((part, i) =>
    re.test(part) && i % 2 === 1 ? (
      <mark key={i} className="rounded bg-accent/25 px-0.5 text-fg">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export function ResultCard({ hit, rank }) {
  return (
    <li className="card p-4">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-xs text-dim">{rank}</span>
        <h3 dir="auto" className="flex-1 font-medium leading-tight">
          {hit.title}
        </h3>
        <span
          className="font-mono text-sm text-accent"
          title="relevance score"
        >
          {hit.score.toFixed(4)}
        </span>
      </div>

      <p dir="auto" className="mt-2 pl-7 text-sm leading-relaxed text-dim">
        {highlight(hit.snippet, hit.matched)}
      </p>

      {hit.matched.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5 pl-7">
          {hit.matched.map((t) => (
            <span
              key={t}
              className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-dim"
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </li>
  );
}

export function RagAnswer({ result }) {
  const cited = new Set(result.citations.map((c) => c.doc_number));

  return (
    <div className="card mb-5 border-accent/40 p-5">
      <div className="mb-3 flex items-center gap-2 text-accent">
        <Sparkle className="h-4 w-4" />
        <span className="text-xs font-medium uppercase tracking-wide">
          Generated answer
        </span>
        {result.model && (
          <span className="ml-auto font-mono text-xs text-dim">
            {result.model}
          </span>
        )}
      </div>

      {result.answer ? (
        // dir="auto" lets the browser's bidi algorithm choose direction from
        // the first strong character — Persian lays out RTL, English LTR, and
        // mixed text (a Persian sentence containing "BM25") stays correct.
        // No language detection, no per-locale branch.
        <p dir="auto" className="whitespace-pre-wrap leading-relaxed">
          {result.answer}
        </p>
      ) : (
        <div>
          <p className="flex items-start gap-2 text-sm text-dim">
            <Alert className="mt-0.5 h-4 w-4 shrink-0" />
            {result.note || "No answer generated."}
          </p>
          {result.coverage.length > 0 && (
            <ul className="mt-3 flex flex-wrap gap-2">
              {result.coverage.map((t) => (
                <li
                  key={t}
                  dir="auto"
                  className="rounded-md border border-line bg-muted px-2 py-1 text-xs"
                >
                  {t}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {result.citations.length > 0 && (
        <div className="mt-4 border-t border-line pt-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-dim">
            Cited sources
          </div>
          <div className="flex flex-wrap gap-2">
            {result.citations.map((c) => (
              <span
                key={c.doc_number}
                className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10 px-2 py-1 text-xs"
              >
                <Doc className="h-3.5 w-3.5 text-accent" />
                <span className="font-mono text-accent">
                  [Doc {c.doc_number}]
                </span>
                <span className="text-fg">{c.title}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {result.hits.length > 0 && (
        <details className="mt-4 border-t border-line pt-3">
          <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-dim hover:text-fg">
            Retrieved passages ({result.hits.length})
          </summary>
          <ul className="mt-3 space-y-2">
            {result.hits.map((h) => (
              <li
                key={h.chunk_id}
                className="flex items-baseline gap-2.5 text-sm"
              >
                <span
                  className={`font-mono text-xs ${
                    cited.has(h.doc_number) ? "text-accent" : "text-dim"
                  }`}
                >
                  [Doc {h.doc_number}]
                </span>
                <span className="flex-1">{h.title}</span>
                <span className="font-mono text-xs text-dim">
                  {h.score.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
