"""Vector Space Model, lnc.ltc — slides 7-Scoring and 8-Scoring.

    documents  lnc : log tf, no idf, cosine normalized   (precomputed at index time)
    queries    ltc : log tf, idf,    cosine normalized

Scoring visits only the postings of query terms, so cost is proportional to
those postings rather than to the collection (slide 8-Scoring s5).

Three retrieval modes, the last two being inexact top-K (slide 8-Scoring s19):

    exact        every chunk containing at least one query term — safe ranking
    champion     only the champion lists, r highest-tf chunks per term (s26)
    elimination  only query terms above an idf floor, all their postings (s24)
"""

import math

from app.core.config import settings
from app.engines.lexical.heap import TopKMinHeap
from app.engines.lexical.index import index, log_tf
from app.engines.lexical.text import tokenize

MODES = ("exact", "champion", "elimination")


def query_vector(query: str) -> dict[str, float]:
    """The ltc query vector: {term: weight}, unit length."""
    tf: dict[str, int] = {}
    for tok in tokenize(query):
        tf[tok] = tf.get(tok, 0) + 1

    weights = {t: log_tf(c) * index.idf(t) for t, c in tf.items() if index.df(t)}
    weights = {t: w for t, w in weights.items() if w > 0}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return {t: w / norm for t, w in weights.items()}


def _eliminate(weights: dict[str, float]) -> dict[str, float]:
    """Index elimination: keep only high-idf query terms — slide 8-Scoring s24.

    "catcher in the rye" scores from catcher and rye only. The floor is a
    fraction of the highest idf in the query, so it adapts to the query rather
    than to a fixed corpus-wide constant, and a query of uniformly common
    terms keeps all of them instead of eliminating itself to nothing.
    """
    if len(weights) < 2:
        return weights
    idfs = {t: index.idf(t) for t in weights}
    floor = settings.elimination_ratio * max(idfs.values())
    kept = {t: w for t, w in weights.items() if idfs[t] >= floor}
    return kept or weights


def search(
    query: str, k: int | None = None, mode: str = "champion", prf: bool = False, **_
) -> dict:
    """Ranked chunks for `query`.

    Every engine returns this shape: {"hits": [...], "scored": int}. `scored`
    is the size of the candidate set actually visited — for the inexact modes
    it is well below the collection size, which is the saving being claimed.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    weights = query_vector(query)
    if mode == "elimination":
        weights = _eliminate(weights)
    if not weights:
        return {"hits": [], "scored": 0}

    k = k or settings.top_k
    if prf:
        from app.engines.lexical import prf as prf_module  # circular at import time

        return prf_module.search(weights, k, mode)
    return rank(weights, k, mode)


def rank(weights: dict[str, float], k: int, mode: str = "exact") -> dict:
    """Cosine-score a query vector against the candidate set and select top-K.

    Split out from `search` because pseudo-relevance feedback (phase 3) ranks a
    modified vector that never came from a query string.
    """
    candidates = index.candidates(weights, mode)
    scores: dict[str, float] = {}
    for cid in candidates:
        forward = index.forward.get(cid)
        if forward is None:
            continue
        norm = index.norm[cid]
        scores[cid] = sum(
            wq * (log_tf(forward[t]) / norm) for t, wq in weights.items() if t in forward
        )

    heap = TopKMinHeap(k)
    for cid, score in scores.items():
        if score > 0:
            heap.push(score, cid)

    matched = set(weights)
    hits = [
        {"chunk_id": cid, "score": score, "matched": matched}
        for score, cid in heap.ranked()
    ]
    return {"hits": hits, "scored": len(candidates)}
