"""Vector Space Model, lnc.ltc — slides 7-Scoring s41/s43 and 8-Scoring.

    documents  lnc : log tf, no idf, cosine normalized   (precomputed at index time)
    queries    ltc : log tf, idf,    cosine normalized

s41 names lnc.ltc "a very standard weighting scheme" and s43 works it through
end to end; this module reproduces that example exactly. Note the document side
carries *no* idf — that is the `n` in lnc, and it is deliberate: idf describes a
term's rarity in the collection, a property of the term rather than of any one
document, so applying it on both sides would square it.

The scored unit is the **document**, per spec §3.3.2 ("a ranked list of relevant
documents") and §3.2.1 ("TF-IDF weighting for queries and documents"). A
document's tf for a term is its total across that document's chunks, and its
`c` normalization is over the whole document — so a long document pays the
length penalty the model intends, rather than being split into several short
ones that each escape it and then compete with each other for result slots.

Scoring visits only the postings of query terms, so cost is proportional to
those postings rather than to the collection (slide 8-Scoring s5).

Three retrieval modes, the last two being inexact top-K (slide 8-Scoring s19):

    exact        every document containing at least one query term — safe
    champion     only the champion lists, r highest-tf documents per term (s26)
    elimination  only query terms above an idf floor, all their postings (s24)
"""

import math

from app.core.config import settings
from app.engines.lexical.heap import TopKMinHeap
from app.engines.lexical.index import index, log_tf
from app.engines.lexical.text import tokenize


def query_vector(query: str) -> dict[str, float]:
    """The ltc query vector: {term: weight}, unit length."""
    tf: dict[str, int] = {}
    for tok in tokenize(query):
        tf[tok] = tf.get(tok, 0) + 1

    weights = {t: log_tf(c) * index.idf(t) for t, c in tf.items() if index.df(t)}
    weights = {t: w for t, w in weights.items() if w > 0}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return {t: w / norm for t, w in weights.items()}


def search(
    query: str, k: int | None = None, mode: str = "champion", prf: bool = False, **_
) -> dict:
    """Ranked documents for `query`.

    Every engine returns this shape: {"hits": [...], "scored": int}. `scored`
    is the size of the candidate set actually visited — for the inexact modes
    it is well below the collection size, which is the saving being claimed.

    `mode` is validated by SearchRequest's Literal at the API boundary, so it
    is trusted here rather than re-checked.
    """
    weights = query_vector(query)
    if mode == "elimination":
        weights = index.eliminate(weights)
    if not weights:
        return {"hits": [], "scored": 0, "contrib": {}, "weights": {}}

    k = k or settings.top_k
    if prf:
        from app.engines.lexical import prf as prf_module  # circular at import time

        return prf_module.search(weights, k, mode)
    return rank(weights, k, mode)


def term_score(wq: float, tf: int, norm: float) -> float:
    """One term's contribution to one document's cosine score.

    The single definition of a cell in the score matrix: the ranking loop sums
    these, and `explain` reports them individually. Two copies of this
    expression would be two things to keep in step, and the heatmap's whole
    claim is that its columns add up to the score already on screen.

    ponytail: a call per (document, term) rather than the expression inlined.
    Inline it if a profile ever shows the call dominating; at this corpus size
    the scoring loop is microseconds either way.
    """
    return wq * (log_tf(tf) / norm)


def explain(weights: dict[str, float], doc_ids: list[str]) -> dict[str, dict[str, float]]:
    """Per-term score breakdown for already-ranked documents.

    Run *after* top-K, over the k survivors only — |q|·k multiplications rather
    than |q|·N. Threading a dict through TopKMinHeap would instead build one per
    candidate and throw away all but k of them, which is exactly the work the
    heap exists to avoid (slide 8-Scoring s13).
    """
    out = {}
    for doc_id in doc_ids:
        forward = index.doc_forward.get(doc_id)
        if forward is None:
            continue
        norm = index.doc_norm[doc_id]
        out[doc_id] = {
            t: term_score(wq, forward[t], norm)
            for t, wq in weights.items()
            if t in forward
        }
    return out


def rank(weights: dict[str, float], k: int, mode: str = "exact") -> dict:
    """Cosine-score a query vector against the candidate documents, top-K.

    Split out from `search` because pseudo-relevance feedback (phase 3) ranks a
    modified vector that never came from a query string.
    """
    candidates = index.candidates(weights, mode)
    heap = TopKMinHeap(k)
    for doc_id in candidates:
        forward = index.doc_forward.get(doc_id)
        if forward is None:
            continue
        norm = index.doc_norm[doc_id]
        score = sum(
            term_score(wq, forward[t], norm)
            for t, wq in weights.items()
            if t in forward
        )
        if score > 0:  # straight into the heap — materializing all N scores
            heap.push(score, doc_id)  # first would defeat the point of O(N log K)

    hits = heap.hits(set(weights))
    return {
        "hits": hits,
        "scored": len(candidates),
        # Keyed by document rather than attached to each hit: Rocchio ranks
        # twice and only the second breakdown describes the list being shown.
        "contrib": explain(weights, [h["doc_id"] for h in hits]),
        # The query vector that actually produced this ranking — after
        # elimination, and after Rocchio replaced it. The heatmap's rows are
        # these terms, not whatever the user typed.
        "weights": dict(weights),
    }
