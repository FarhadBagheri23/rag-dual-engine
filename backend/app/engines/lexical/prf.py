"""Pseudo-relevance feedback via Rocchio — slides 10-Relevance Feedback.

    q_m = α·q_0 + β·(1/|D_r|)Σ_{d∈D_r} d − γ·(1/|D_nr|)Σ_{d∈D_nr} d

True relevance feedback needs a user to judge documents. Pseudo-relevance
feedback assumes the top-k of an initial retrieval are relevant and the next
few are not (s21), then runs Rocchio on that assumption and retrieves again.

Two details from the slides that are easy to miss:

  * Negative term weights are meaningless in the vector space model, so they
    are clipped to zero (s12).
  * Rocchio produces a very long query, and long queries are expensive to
    process — the slides cap the expansion at the top ~20 terms (s20).

The centroid is built from *ltc* document vectors, not the *lnc* weights the
index stores. Rocchio adds documents to a query, so both must live in the
same space, and the query side is idf-weighted. The feedback set is only a
few dozen chunks, so recomputing their idf-weighted vectors is cheap.
"""

import math

from app.core.config import settings
from app.engines.lexical.index import index, log_tf
from app.engines.lexical.vsm import rank


def _doc_vector_ltc(chunk_id: str) -> dict[str, float]:
    """A chunk as a unit-length idf-weighted vector — the query's space."""
    forward = index.forward.get(chunk_id, {})
    vec = {t: log_tf(tf) * index.idf(t) for t, tf in forward.items()}
    vec = {t: w for t, w in vec.items() if w > 0}
    norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
    return {t: w / norm for t, w in vec.items()}


def _centroid(chunk_ids: list[str]) -> dict[str, float]:
    if not chunk_ids:
        return {}
    acc: dict[str, float] = {}
    for cid in chunk_ids:
        for term, weight in _doc_vector_ltc(cid).items():
            acc[term] = acc.get(term, 0.0) + weight
    n = len(chunk_ids)
    return {t: w / n for t, w in acc.items()}


def expand(q0: dict[str, float], ranked: list[dict]) -> dict[str, float]:
    """One Rocchio iteration over an initial ranking. Returns the new query."""
    ids = [h["chunk_id"] for h in ranked]
    relevant = ids[: settings.prf_n_relevant]
    nonrelevant = ids[
        settings.prf_n_relevant : settings.prf_n_relevant + settings.prf_n_nonrelevant
    ]

    c_rel, c_non = _centroid(relevant), _centroid(nonrelevant)

    qm: dict[str, float] = {}
    for term, weight in q0.items():
        qm[term] = qm.get(term, 0.0) + settings.prf_alpha * weight
    for term, weight in c_rel.items():
        qm[term] = qm.get(term, 0.0) + settings.prf_beta * weight
    for term, weight in c_non.items():
        qm[term] = qm.get(term, 0.0) - settings.prf_gamma * weight

    qm = {t: w for t, w in qm.items() if w > 0}  # clip negatives (s12)

    # keep the heaviest terms only (s20) — but never drop an original query
    # term, which α is there to protect
    if len(qm) > settings.prf_expansion_terms:
        keep = sorted(qm, key=qm.get, reverse=True)[: settings.prf_expansion_terms]
        qm = {t: w for t, w in qm.items() if t in keep or t in q0}

    norm = math.sqrt(sum(w * w for w in qm.values())) or 1.0
    return {t: w / norm for t, w in qm.items()}


def search(q0: dict[str, float], k: int, mode: str) -> dict:
    """Retrieve, expand, retrieve again."""
    first = rank(q0, settings.prf_n_relevant + settings.prf_n_nonrelevant, mode)
    if not first["hits"]:
        return {"hits": [], "scored": first["scored"], "expansion": []}

    qm = expand(q0, first["hits"])
    second = rank(qm, k, mode)

    added = sorted(set(qm) - set(q0), key=qm.get, reverse=True)
    return {
        "hits": second["hits"],
        "scored": first["scored"] + second["scored"],
        "expansion": added,
    }
