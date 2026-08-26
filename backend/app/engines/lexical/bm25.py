"""Okapi BM25 — slides 11-Probabilistic, s29-32.

                    (k1 + 1) tf_td              (k3 + 1) qf_t
    score = Σ idf_t ────────────────────────  ─────────────────
            t∈q     k1(1 - b + b·dl/avgdl)     k3 + qf_t
                       + tf_td

Three parts, each doing one job:
    idf_t   rare terms count for more, as in the vector space model
    k1, b   term frequency saturates, and long documents are normalized —
            the two things the Binary Independence Model lacked (s29)
    k3      the same saturation applied to the query side. It only matters
            for long queries where a term repeats (s32, "Weighting for Long
            Queries"); for typical short queries qf=1 and the factor is
            constant across documents, so it cannot change the ranking.
"""

import math

from app.core.config import settings
from app.engines.lexical.heap import TopKMinHeap
from app.engines.lexical.index import index
from app.engines.lexical.text import tokenize
from app.engines.lexical.vsm import MODES


def idf(term: str) -> float:
    """log(N/df) — the whole-collection approximation from slide 11 s22.

    Matches the course's own homework. The Robertson-Sparck Jones form,
    log((N - df + 0.5)/(df + 0.5)), is the other common choice; it goes
    negative for terms in more than half the collection, which the simpler
    form never does.
    """
    d = index.df(term)
    return math.log(index.n_chunks / d) if d else 0.0


def search(
    query: str, k: int | None = None, mode: str = "champion", prf: bool = False
) -> dict:
    """Ranked chunks for `query`. Same contract as vsm.search.

    `prf` is accepted so every engine shares one signature; Rocchio is defined
    on the vector space model, and the route rejects prf with this engine.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    qf: dict[str, int] = {}
    for tok in tokenize(query):
        if index.df(tok):
            qf[tok] = qf.get(tok, 0) + 1
    if not qf:
        return {"hits": [], "scored": 0}

    if mode == "elimination":
        floor = settings.elimination_ratio * max(idf(t) for t in qf)
        qf = {t: c for t, c in qf.items() if idf(t) >= floor} or qf

    k1, b, k3 = settings.bm25_k1, settings.bm25_b, settings.bm25_k3
    avgdl = index.avg_len or 1.0

    candidates = index.candidates(qf, mode)
    heap = TopKMinHeap(k or settings.top_k)
    for cid in candidates:
        forward = index.forward.get(cid)
        if forward is None:
            continue
        norm_len = 1 - b + b * (index.doc_len[cid] / avgdl)
        score = 0.0
        for term, count in qf.items():
            tf = forward.get(term)
            if not tf:
                continue
            score += (
                idf(term)
                * ((k1 + 1) * tf) / (k1 * norm_len + tf)
                * ((k3 + 1) * count) / (k3 + count)
            )
        if score > 0:
            heap.push(score, cid)

    matched = set(qf)
    hits = [
        {"chunk_id": cid, "score": score, "matched": matched}
        for score, cid in heap.ranked()
    ]
    return {"hits": hits, "scored": len(candidates)}
