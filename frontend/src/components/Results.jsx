/** Wrap every matched term in the snippet. Terms come from the backend's own
 *  tokenizer ([^\W_]+), so they are already regex-safe.
 *
 *  The boundaries are Unicode property lookarounds, not \b: JS defines \b over
 *  [A-Za-z0-9_] only, so /\b(بازیابی)\b/ never matches — Persian, Arabic and
 *  CJK terms went silently unhighlighted even though the backend indexes them.
 *  \p{L}\p{N} with the u flag is the boundary \b should have been.
 *
 *  split() with one capture group alternates [text, match, text, match, …], so
 *  an odd index IS the match — no second test needed (and re.test on a /g
 *  regex is stateful, which is a trap for whoever edits this next).
 *
 *  The trailing characters are captured rather than merely permitted, because
 *  `terms` are Porter stems: the backend matched "retrieval" on the stem
 *  "retriev", so a marker that stopped at the stem would leave "al" outside the
 *  highlight. Consuming the rest of the word marks the surface form the reader
 *  actually sees. */
function highlight(text, terms) {
  if (!terms?.length) return text;
  const re = new RegExp(
    `(?<![\\p{L}\\p{N}])(${terms.join("|")}[\\p{L}\\p{N}]*)`,
    "giu"
  );
  return text.split(re).map((part, i) =>
    i % 2 === 1 ? (
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
