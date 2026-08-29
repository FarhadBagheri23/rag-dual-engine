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
        return {"hits": [], "scored": 0}

    if mode == "elimination":
        qf = index.eliminate(qf)

    k1, b, k3 = settings.bm25_k1, settings.bm25_b, settings.bm25_k3
    avgdl = index.avg_len or 1.0

    candidates = index.candidates(qf, mode)
    heap = TopKMinHeap(k or settings.top_k)
    for doc_id in candidates:
        forward = index.doc_forward.get(doc_id)
        if forward is None:
            continue
        norm_len = 1 - b + b * (index.doc_len[doc_id] / avgdl)
        score = 0.0
        for term, count in qf.items():
            tf = forward.get(term)
            if not tf:
                continue
            score += (
                index.idf(term)
                * ((k1 + 1) * tf) / (k1 * norm_len + tf)
                * ((k3 + 1) * count) / (k3 + count)
            )
        if score > 0:
            heap.push(score, doc_id)

    return {"hits": heap.hits(set(qf)), "scored": len(candidates)}
