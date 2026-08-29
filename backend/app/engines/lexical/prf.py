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

The feedback set is **documents**, per spec §3.2.1: "assume the top k documents
are relevant, expand the query with the most frequent/important terms from these
documents". Building the centroid from chunks instead would let one long
document contribute several times over, weighting the expansion by how the
corpus happened to be split.

The centroid is built from *ltc* document vectors, not the *lnc* weights the
index stores. Rocchio adds documents to a query, so both must live in the
same space, and the query side is idf-weighted. The feedback set is only a
few dozen documents, so recomputing their idf-weighted vectors is cheap.
"""

import math

from app.core.config import settings
from app.engines.lexical.index import index, log_tf
from app.engines.lexical.vsm import rank


def _doc_vector_ltc(doc_id: str) -> dict[str, float]:
    """A document as a unit-length idf-weighted vector — the query's space."""
    forward = index.doc_forward.get(doc_id, {})
    vec = {t: log_tf(tf) * index.idf(t) for t, tf in forward.items()}
    vec = {t: w for t, w in vec.items() if w > 0}
    norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
    return {t: w / norm for t, w in vec.items()}


def _centroid(doc_ids: list[str]) -> dict[str, float]:
    if not doc_ids:
        return {}
    acc: dict[str, float] = {}
    for doc_id in doc_ids:
        for term, weight in _doc_vector_ltc(doc_id).items():
            acc[term] = acc.get(term, 0.0) + weight
    n = len(doc_ids)
    return {t: w / n for t, w in acc.items()}


def expand(q0: dict[str, float], ranked: list[dict]) -> dict[str, float]:
    """One Rocchio iteration over an initial ranking. Returns the new query."""
    ids = [h["doc_id"] for h in ranked]
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
        # Total scoring work, summed over both retrievals — which can exceed the
        # collection size, because a document scored in both passes is scored
        # twice. `passes` carries the split so the UI can say "6 + 7 of 9"
        # rather than the nonsense "13 of 9".
        "scored": first["scored"] + second["scored"],
        "passes": [first["scored"], second["scored"]],
        "expansion": added,
    }
