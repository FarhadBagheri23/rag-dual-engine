/** The score matrix, drawn.
 *
 *  Rows are the query terms the engine actually ranked on, columns are the
 *  ranked documents, and a cell is what that term contributed to that
 *  document's score. This is slide 7-Scoring s29 — "Binary → Count → Weight
 *  Matrix" — rendered on the live corpus instead of the slide's three sonnets.
 *
 *  The claim that makes it worth drawing: **every column sums to the score
 *  already printed beside that result**. So the picture is checkable, not
 *  decorative, and it answers the question a ranked list never does — *why*
 *  did #1 beat #3, and which term carried it.
 *
 *  Colour is sequential: one hue per engine, alpha rising with magnitude, so
 *  more is brighter against the dark surface. Cells carry no number. In the
 *  middle of the ramp no ink clears 4.5:1 against the blend (measured: 4.19
 *  for sky, 4.30 for amber), and a number on every cell is noise besides —
 *  only the heaviest cell in each column is labelled, where the fill is at
 *  full alpha and contrast reaches 10.7:1. Every value is still reachable: the
 *  cell's native tooltip carries it, and the footer carries the column total.
 */

// rgb triples rather than the Tailwind hex tokens, because a sequential ramp
// needs a per-cell alpha and Tailwind's arbitrary-value syntax cannot take a
// runtime number. Same three colours as tailwind.config.js, which is where the
// contrast for each was established.
const HUE = {
  vsm: [125, 211, 252], // sky   — matches the vsm tag in the history sidebar
  bm25: [252, 211, 77], // amber — matches the bm25 tag
};

const SURFACE = "#1B2336"; // bg-card, the colour cells are blended against
const INK = "#0F172A"; // accent-ink, the label colour on a bright fill

/** Fraction of the maximum above which a cell may carry a printed number.
 *
 *  Measured, not guessed. Blending each hue over #1B2336 and running WCAG both
 *  ways: light ink loses to dark ink around alpha 0.52 (amber) / 0.56 (sky),
 *  and in that band *neither* reaches 4.5:1 — 4.30 and 4.19 at worst. Alpha
 *  0.60 is the first step where dark ink clears it (4.95 sky, 5.48 amber), and
 *  alpha = 0.12 + 0.88·f puts that at f = 0.55.
 *
 *  Below it the cell simply carries no number, which is the better graphic
 *  anyway — a number on every cell is noise, and the value is still one hover
 *  away. */
const LABEL_FLOOR = 0.55;

/** Magnitude → fill. A floor of 0.12 so the weakest real contribution is still
 *  visibly a contribution, distinct from the "term absent" cell below it. */
const alphaFor = (v, max) => (max > 0 ? 0.12 + 0.88 * (v / max) : 0);

const fill = (engine, v, max) => {
  const [r, g, b] = HUE[engine] ?? HUE.vsm;
  return `rgba(${r}, ${g}, ${b}, ${alphaFor(v, max)})`;
};

export default function Heatmap({ engine, terms, hits, compact = false }) {
  // Rows come from the response rather than from the hits: a term that matched
  // nothing appears in no column, and would silently vanish from a grid built
  // by scanning the results. An empty row is the most useful row on the page —
  // it says "this word of your query found nothing" at a glance.
  if (!terms?.length || !hits?.length) return null;

  const value = (hit, term) => hit.contrib?.[term] ?? 0;
  const max = Math.max(
    ...hits.flatMap((h) => terms.map((t) => value(h, t))),
    0
  );
  if (max <= 0) return null; // dense retrieval: no per-term decomposition

  // The heaviest cell per column is the one that gets a printed number.
  const peak = hits.map((h) =>
    terms.reduce((best, t) => (value(h, t) > value(h, best) ? t : best), terms[0])
  );

  const cell = compact ? "h-7 w-11" : "h-9 w-14";

  return (
    <section
      aria-labelledby={`hm-${engine}`}
      className="card overflow-hidden"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <h3 id={`hm-${engine}`} className="text-sm font-medium">
            Score matrix
            <span className="ml-2 font-mono text-xs uppercase text-dim">
              {engine}
            </span>
          </h3>
          <p className="mt-0.5 text-xs text-dim">
            each column sums to that result’s score
          </p>
        </div>
        <Legend engine={engine} max={max} />
      </header>

      {/* Wide grids scroll inside their own box; the page never scrolls
          sideways. */}
      <div className="overflow-x-auto px-4 py-3.5">
        <table className="border-separate border-spacing-0.5">
          <caption className="sr-only">
            Query terms by ranked document. Each cell is that term’s
            contribution to that document’s relevance score.
          </caption>
          <thead>
            <tr>
              <th className="sr-only" scope="col">
                Query term
              </th>
              {hits.map((h, i) => (
                <th
                  key={h.doc_id}
                  scope="col"
                  title={h.title}
                  className="pb-1 text-center font-mono text-[11px] font-normal text-dim"
                >
                  {i + 1}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {terms.map((term) => {
              const dead = hits.every((h) => value(h, term) === 0);
              return (
                <tr key={term}>
                  <th
                    scope="row"
                    className={`whitespace-nowrap pr-3 text-right font-mono text-xs font-normal ${
                      dead ? "text-dim/50 line-through" : "text-dim"
                    }`}
                    title={
                      dead
                        ? `“${term}” matched none of these documents`
                        : `query term “${term}”`
                    }
                  >
                    {term}
                  </th>

                  {hits.map((h, i) => {
                    const v = value(h, term);
                    const label = `${term} → result ${i + 1} (${h.title}): ${
                      v > 0 ? v.toFixed(4) : "term absent"
                    }`;
                    return (
                      <td key={h.doc_id} className="p-0">
                        <div
                          title={label}
                          aria-label={label}
                          className={`${cell} flex items-center justify-center rounded transition-shadow duration-150 hover:ring-2 hover:ring-fg/40 ${
                            v > 0 ? "" : "bg-muted/40"
                          }`}
                          style={
                            v > 0
                              ? {
                                  backgroundColor: fill(engine, v, max),
                                  // The 2px grid gap reads as a gap only if the
                                  // fill stops there; without this the rounded
                                  // corners sit on the page background.
                                  boxShadow: `0 0 0 1px ${SURFACE}`,
                                }
                              : undefined
                          }
                        >
                          {peak[i] === term && v / max >= LABEL_FLOOR && (
                            <span
                              className="font-mono text-[10px] font-semibold tabular-nums"
                              style={{ color: INK }}
                            >
                              {v.toFixed(2)}
                            </span>
                          )}
                          {v === 0 && (
                            <span className="text-dim/40" aria-hidden="true">
                              ·
                            </span>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>

          {/* The verification row. If these do not equal the scores in the
              result list, the decomposition is wrong — which is the point of
              printing them here rather than trusting the picture. */}
          <tfoot>
            <tr>
              <th
                scope="row"
                className="pr-3 pt-1.5 text-right font-mono text-[10px] font-normal uppercase tracking-wide text-dim"
              >
                score
              </th>
              {hits.map((h) => (
                <td
                  key={h.doc_id}
                  className="pt-1.5 text-center font-mono text-[10px] tabular-nums text-fg"
                >
                  {h.score.toFixed(3)}
                </td>
              ))}
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}

/** Sequential scale key. Five steps is enough to read the direction; the
 *  endpoints are labelled because an unlabelled ramp says nothing about
 *  magnitude. */
function Legend({ engine, max }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] text-dim">0</span>
      <div className="flex gap-px" aria-hidden="true">
        {[0.2, 0.4, 0.6, 0.8, 1].map((f) => (
          <span
            key={f}
            className="h-3 w-4 first:rounded-l last:rounded-r"
            style={{ backgroundColor: fill(engine, f * max, max) }}
          />
        ))}
      </div>
      <span className="font-mono text-[10px] tabular-nums text-dim">
        {max.toFixed(2)}
      </span>
    </div>
  );
}
