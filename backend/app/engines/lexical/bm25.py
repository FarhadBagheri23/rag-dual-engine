"""Okapi BM25 — slides 11-Probabilistic, s29-32.

                    (k1 + 1) tf_td              (k3 + 1) qf_t
    score = Σ idf_t ────────────────────────  ─────────────────
            t∈q     k1(1 - b + b·dl/avgdl)     k3 + qf_t
                       + tf_td

Three parts, each doing one job:
    idf_t   rare terms count for more, as in the vector space model. s30-31
            write it as log(N/df) — the same whole-collection approximation
            (s22) the VSM uses, so this engine calls `index.idf` rather than
            defining a second one. Neither slide pins a base, and the base only
            scales every score by a constant, so sharing one definition costs
            nothing and makes df and idf mean the same thing in both engines.
            The Robertson-Sparck Jones form, log((N - df + 0.5)/(df + 0.5)), is
            the other common choice; it goes negative for terms in more than
            half the collection, which the simpler form never does.
    k1, b   term frequency saturates, and long documents are normalized —
            the two things the Binary Independence Model lacked (s29). Both
            dl and avgdl are measured over *documents*: chunking makes every
            chunk roughly one length by construction, so normalizing by chunk
            length would correct for a variance that chunking already removed
            and leave b with almost nothing to do.
    k3      the same saturation applied to the query side. It only matters
            for long queries where a term repeats (s32, "Weighting for Long
            Queries"); for typical short queries qf=1 and the factor is
            constant across documents, so it cannot change the ranking.

s32 sets the defaults this uses: k1 and k3 between 1.2 and 2, b = 0.75.
"""

from app.core.config import settings
from app.engines.lexical.heap import TopKMinHeap
from app.engines.lexical.index import index
from app.engines.lexical.text import tokenize


def term_score(term: str, tf: int, norm_len: float, qf_t: int) -> float:
    """One term's contribution to one document's BM25 score — the whole summand.

    The single definition of a cell in the score matrix: the ranking loop sums
    these and `explain` reports them individually, so the heatmap's columns add
    up to exactly the score shown beside each result. See vsm.term_score for the
    same argument on the other engine.
    """
    k1, b, k3 = settings.bm25_k1, settings.bm25_b, settings.bm25_k3
    return (
        index.idf(term)
        * ((k1 + 1) * tf) / (k1 * norm_len + tf)
        * ((k3 + 1) * qf_t) / (k3 + qf_t)
    )


def _norm_len(doc_id: str, b: float, avgdl: float) -> float:
    """The `1 - b + b·dl/avgdl` length factor, over documents (see module docstring)."""
    return 1 - b + b * (index.doc_len[doc_id] / avgdl)


def explain(qf: dict[str, int], doc_ids: list[str]) -> dict[str, dict[str, float]]:
    """Per-term score breakdown for already-ranked documents — |q|·k work.

    Runs after top-K for the same reason vsm.explain does: building a breakdown
    per candidate would allocate for every document the heap is about to reject.
    """
    avgdl = index.avg_len or 1.0
    out = {}
    for doc_id in doc_ids:
        forward = index.doc_forward.get(doc_id)
        if forward is None:
            continue
        norm_len = _norm_len(doc_id, settings.bm25_b, avgdl)
        out[doc_id] = {
            term: term_score(term, forward[term], norm_len, count)
            for term, count in qf.items()
            if forward.get(term)
        }
    return out


def search(
    query: str, k: int | None = None, mode: str = "champion", prf: bool = False, **_
) -> dict:
    """Ranked documents for `query`. Same contract as vsm.search.

    `prf` is accepted so every engine shares one signature; Rocchio is defined
    on the vector space model, and the route rejects prf with this engine.
    `**_` absorbs options that belong to another engine, such as `model`.

    `mode` is validated by SearchRequest's Literal at the API boundary, so it
    is trusted here rather than re-checked. Note that `champion` is a looser
    approximation for BM25 than for the vector space model: s26 scopes champion
    lists to tf-idf, where score rises monotonically with tf, whereas BM25 also
    divides by document length — so a high-tf long document can rank below a
    low-tf short one that the tf-ordered list left out. Both modes are non-safe
    by construction (s16); this one is merely less tight.
    """
    # idf > 0, not merely df > 0: a term in every document scores exactly zero,
    # so keeping it only widens the candidate set and inflates `scored` — the
    # very number the inexact top-K modes are measured by. vsm.query_vector
    # applies the same filter, so both engines report comparable counts.
    qf: dict[str, int] = {}
    for tok in tokenize(query):
        if index.idf(tok) > 0:
            qf[tok] = qf.get(tok, 0) + 1
    if not qf:
        return {"hits": [], "scored": 0, "contrib": {}, "weights": {}}

    if mode == "elimination":
        qf = index.eliminate(qf)

    b = settings.bm25_b
    avgdl = index.avg_len or 1.0

    candidates = index.candidates(qf, mode)
    heap = TopKMinHeap(k or settings.top_k)
    for doc_id in candidates:
        forward = index.doc_forward.get(doc_id)
        if forward is None:
            continue
        norm_len = _norm_len(doc_id, b, avgdl)
        score = 0.0
        for term, count in qf.items():
            tf = forward.get(term)
            if not tf:
                continue
            score += term_score(term, tf, norm_len, count)
        if score > 0:
            heap.push(score, doc_id)

    hits = heap.hits(set(qf))
    return {
        "hits": hits,
        "scored": len(candidates),
        "contrib": explain(qf, [h["doc_id"] for h in hits]),
        # The query side of the summand — everything in the formula that does
        # not depend on the document. Raw qf would do to say *which* terms were
        # searched on, but not to order them: for a normal query every qf is 1,
        # and the heatmap's rows would come out in whatever order the tokenizer
        # happened to emit. Ordered by this, the rarest term is the top row in
        # both engines, and the two matrices can be read against each other.
        "weights": {
            t: index.idf(t)
            * ((settings.bm25_k3 + 1) * c)
            / (settings.bm25_k3 + c)
            for t, c in qf.items()
        },
    }
