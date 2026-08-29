import { useEffect, useState } from "react";
import * as api from "../api";
import Heatmap from "./Heatmap";
import { Alert, Spinner } from "./icons";

/** One query, both lexical engines, side by side.
 *
 *  Spec §7 asks for "identical queries across VSM, BM25 and RAG to compare the
 *  results". A ranked list each does not actually let you compare them — you
 *  read two lists of titles and take it on faith that they differ. What this
 *  section shows instead is *where* they differ and *why*:
 *
 *    · agreement@k — how much of the top-k both engines chose (Jaccard)
 *    · rank shift  — per document, where it moved and by how much
 *    · two score matrices — the term-level reason it moved
 *
 *  The heatmaps are the payoff. BM25's tf factor saturates at (k1+1) while the
 *  vector space model's 1 + log tf has no ceiling, so a term that swallows a
 *  whole column under VSM flattens under BM25 and leaves room for the rest of
 *  the query. That is slides 7-Scoring s23 against 11-Probabilistic s29-31,
 *  visible on the reader's own corpus rather than described.
 *
 *  ponytail: two calls to the existing /api/search, differenced in the
 *  browser. No compare endpoint — the server has nothing to add that the two
 *  responses do not already carry, and a third route would be a third thing
 *  to keep in step with the engines.
 */
export default function Compare({ query, mode, docCount, onClose }) {
  const [state, setState] = useState({ busy: true });

  useEffect(() => {
    let live = true;
    setState({ busy: true });
    Promise.all(
      ["vsm", "bm25"].map((engine) =>
        api.search({ query, engine, mode, prf: false })
      )
    )
      .then(([vsm, bm25]) => live && setState({ busy: false, vsm, bm25 }))
      .catch((e) => live && setState({ busy: false, error: e.message }));
    return () => {
      live = false; // a fast re-query must not have its result overwritten by a slow one
    };
  }, [query, mode]);

  const { busy, error, vsm, bm25 } = state;

  return (
    <section className="mt-8" aria-labelledby="cmp-h">
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 id="cmp-h" className="text-base font-medium">
            VSM vs BM25
          </h2>
          <p className="mt-0.5 text-sm text-dim">
            Both engines, the same query, the same candidate set.
          </p>
        </div>
        <button
          onClick={onClose}
          className="cursor-pointer rounded-md border border-line px-3 py-1.5 text-sm text-dim transition-colors duration-150 hover:border-edge hover:text-fg"
        >
          Close
        </button>
      </header>

      {busy && (
        <div className="card flex items-center gap-3 p-8 text-sm text-dim">
          <Spinner className="h-4 w-4" />
          Running both engines…
        </div>
      )}

      {error && (
        <div className="card flex items-start gap-2.5 border-danger/50 p-4 text-sm">
          <Alert className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
          <span>{error}</span>
        </div>
      )}

      {!busy && !error && vsm && bm25 && (
        <Report vsm={vsm} bm25={bm25} docCount={docCount} />
      )}
    </section>
  );
}

function Report({ vsm, bm25, docCount }) {
  const a = vsm.hits.map((h) => h.doc_id);
  const b = bm25.hits.map((h) => h.doc_id);
  const shared = a.filter((id) => b.includes(id));
  const union = new Set([...a, ...b]);

  if (union.size === 0) {
    return (
      <div className="card p-8 text-center text-sm text-dim">
        Neither engine matched this query.
      </div>
    );
  }

  // Agreement over the *union*, not over k: with 3 hits each and 3 shared,
  // "3 of 10" would read as disagreement when the two rankings are identical.
  const agreement = shared.length / union.size;

  // Rank movement per document, VSM's order as the reference frame.
  const rows = [...union].map((id) => {
    const hit = vsm.hits.find((h) => h.doc_id === id) ?? bm25.hits.find((h) => h.doc_id === id);
    const ra = a.indexOf(id);
    const rb = b.indexOf(id);
    return { id, title: hit.title, vsm: ra < 0 ? null : ra + 1, bm25: rb < 0 ? null : rb + 1 };
  });
  rows.sort((x, y) => (x.vsm ?? 99) - (y.vsm ?? 99));
  // Identical rankings are a real and common outcome on a small corpus, and
  // the panel has to say so. A table of dashes under "where they disagree"
  // reads as a broken feature rather than as the finding it is.
  const moved = rows.filter((r) => r.vsm !== r.bm25).length;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="agreement@k"
          value={`${Math.round(agreement * 100)}%`}
          note={`${shared.length} of ${union.size} documents chosen by both`}
        />
        <Stat
          label="documents scored"
          value={`${vsm.scored} / ${bm25.scored}`}
          note={`of ${docCount} — VSM / BM25, same inexact top-K mode`}
        />
        <Stat
          label="latency"
          value={`${vsm.took_ms.toFixed(1)} / ${bm25.took_ms.toFixed(1)} ms`}
          note="server-side scoring only, excludes the network"
        />
      </div>

      {/* Ranks, not scores. VSM's cosine lives in [0,1] and BM25's sum of idf
          terms does not — comparing the two numbers directly is meaningless,
          and only the order they induce can be held side by side. */}
      <div className="card overflow-hidden">
        <header className="border-b border-line px-4 py-3">
          <h3 className="text-sm font-medium">
            {moved === 0
              ? "Both engines ranked these documents identically"
              : `Where the rankings disagree — ${moved} document${
                  moved > 1 ? "s" : ""
                } moved`}
          </h3>
          <p className="mt-0.5 text-xs text-dim">
            {moved === 0
              ? "Same order, different arithmetic — compare the two matrices below to see the scores that produced it."
              : "Positions, not scores — a cosine and a BM25 sum are not on one scale."}
          </p>
        </header>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-dim">
                <th className="px-4 py-2 font-medium">Document</th>
                <th className="w-20 px-4 py-2 text-right font-medium">VSM</th>
                <th className="w-20 px-4 py-2 text-right font-medium">BM25</th>
                <th className="w-24 px-4 py-2 text-right font-medium">Move</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-b border-line/60 last:border-0">
                  <td dir="auto" className="max-w-0 truncate px-4 py-2" title={r.title}>
                    {r.title}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums text-vsm">
                    {r.vsm ?? "—"}
                  </td>
                  <td className="px-4 py-2 text-right font-mono tabular-nums text-bm25">
                    {r.bm25 ?? "—"}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <Move from={r.vsm} to={r.bm25} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Heatmap engine="vsm" terms={vsm.terms} hits={vsm.hits} compact />
        <Heatmap engine="bm25" terms={bm25.terms} hits={bm25.hits} compact />
      </div>

      <p className="text-xs leading-relaxed text-dim">
        Column order differs between the two matrices — each is in its own
        engine’s rank order, which is the thing being compared. Read a row
        across to see one term’s pull, and compare the same term between the two
        panels to see BM25’s term-frequency saturation flatten what the vector
        space model let run.
      </p>
    </div>
  );
}

function Stat({ label, value, note }) {
  return (
    <div className="card p-4">
      <div className="text-xs uppercase tracking-wide text-dim">{label}</div>
      <div className="mt-1 font-mono text-xl tabular-nums">{value}</div>
      <div className="mt-1 text-xs leading-snug text-dim">{note}</div>
    </div>
  );
}

/** Rank movement. The arrow is redundant with the sign and the colour, on
 *  purpose — colour alone would carry the meaning for nobody who cannot see it. */
function Move({ from, to }) {
  if (from == null)
    return <span className="font-mono text-xs text-bm25">BM25 only</span>;
  if (to == null)
    return <span className="font-mono text-xs text-vsm">VSM only</span>;
  const d = from - to; // positive: BM25 ranks it higher
  if (d === 0) return <span className="font-mono text-xs text-dim">—</span>;
  return (
    <span
      className={`font-mono text-xs tabular-nums ${d > 0 ? "text-accent" : "text-danger"}`}
      title={`BM25 ranks this ${Math.abs(d)} place${Math.abs(d) > 1 ? "s" : ""} ${
        d > 0 ? "higher" : "lower"
      } than VSM`}
    >
      {d > 0 ? "▲" : "▼"} {Math.abs(d)}
    </span>
  );
}
