"""Phase 3 checks: BM25 against values computed from the slide formula, and
Rocchio pseudo-relevance feedback actually changing the query.

Run with:  cd backend && python -m tests.test_bm25_prf
"""

import math

from app.core.config import settings
from app.database.session import init_db
from app.engines.lexical import bm25, prf, vsm
from app.engines.lexical.index import index
from tests.test_lexical import load_toy


def test_bm25_against_hand_computation():
    """Query 'insurance' over the toy corpus, scored from the slide formula.

        c1 "car insurance auto insurance"  tf=2  dl=4
        c2 "auto insurance"                tf=1  dl=2
        N=3, df(insurance)=2, avgdl = (4+2+2)/3 = 2.6667
    """
    load_toy()
    k1, b, k3 = settings.bm25_k1, settings.bm25_b, settings.bm25_k3
    assert (k1, b, k3) == (1.5, 0.75, 1.2), (k1, b, k3)

    avgdl = (4 + 2 + 2) / 3
    idf = math.log(3 / 2)  # natural log, whole-collection approximation

    def expect(tf, dl, qf=1):
        norm_len = 1 - b + b * (dl / avgdl)
        return idf * ((k1 + 1) * tf) / (k1 * norm_len + tf) * ((k3 + 1) * qf) / (k3 + qf)

    result = bm25.search("insurance", k=10, mode="exact")
    got = {h["chunk_id"]: h["score"] for h in result["hits"]}

    assert math.isclose(got["c1"], expect(tf=2, dl=4), abs_tol=1e-9), got
    assert math.isclose(got["c2"], expect(tf=1, dl=2), abs_tol=1e-9), got
    assert math.isclose(index.avg_len, avgdl, rel_tol=1e-12)
    print(f"  c1={got['c1']:.7f} (expected {expect(2, 4):.7f})")
    print(f"  c2={got['c2']:.7f} (expected {expect(1, 2):.7f})")


def test_bm25_and_vsm_disagree_on_length_penalty():
    """Same query, opposite winners — the two models normalize length
    differently. Cosine divides by the full vector norm, which punishes c1
    hard for being longer; BM25's b=0.75 is gentler, so c1's higher term
    frequency wins. Worth demonstrating: it is why the spec asks for both."""
    load_toy()
    vsm_top = vsm.search("insurance", k=10, mode="exact")["hits"][0]["chunk_id"]
    bm25_top = bm25.search("insurance", k=10, mode="exact")["hits"][0]["chunk_id"]
    assert vsm_top == "c2" and bm25_top == "c1", (vsm_top, bm25_top)
    print(f"  vsm top={vsm_top}  bm25 top={bm25_top}  (same query, different model)")


def test_k3_only_matters_for_repeated_query_terms():
    """Slide 11 s32 calls k3 the long-query weighting, and the algebra says
    exactly how long: with qf=1 the factor is (k3+1)·1/(k3+1) = 1 for every
    value of k3, so it does not merely fail to reorder — it vanishes. It only
    becomes a real term once a query word repeats."""
    load_toy()
    original = settings.bm25_k3
    try:
        no_repeat_a = bm25.search("car insurance", k=10, mode="exact")["hits"]
        repeat_a = bm25.search("insurance insurance", k=10, mode="exact")["hits"]

        settings.bm25_k3 = 50.0
        no_repeat_b = bm25.search("car insurance", k=10, mode="exact")["hits"]
        repeat_b = bm25.search("insurance insurance", k=10, mode="exact")["hits"]

        # qf=1 everywhere: unchanged to floating-point precision. (Not bit
        # identical — (2.2*1)/2.2 does not land on exactly 1.0 — but the
        # residual is ~1e-16, i.e. the factor really is inert.)
        for x, y in zip(no_repeat_a, no_repeat_b):
            assert x["chunk_id"] == y["chunk_id"]
            assert math.isclose(x["score"], y["score"], rel_tol=1e-12), (x, y)
        # qf=2: the factor is live, so scores move by a visible amount
        assert not math.isclose(
            repeat_a[0]["score"], repeat_b[0]["score"], rel_tol=1e-6
        ), (repeat_a[0], repeat_b[0])
        print(f"  qf=1: {no_repeat_a[0]['score']:.7f} -> {no_repeat_b[0]['score']:.7f} "
              f"(k3 inert)")
        print(f"  qf=2: {repeat_a[0]['score']:.4f} -> {repeat_b[0]['score']:.4f} "
              f"(k3 live)")
    finally:
        settings.bm25_k3 = original


def test_rocchio_expands_and_stays_valid():
    load_toy()
    saved = (settings.prf_n_relevant, settings.prf_n_nonrelevant)
    try:
        settings.prf_n_relevant, settings.prf_n_nonrelevant = 1, 1

        q0 = vsm.query_vector("insurance")
        first = vsm.rank(q0, 2, "exact")
        qm = prf.expand(q0, first["hits"])

        assert set(qm) > set(q0), (sorted(q0), sorted(qm))
        assert all(w > 0 for w in qm.values()), "negative weights not clipped"
        assert math.isclose(
            math.sqrt(sum(w * w for w in qm.values())), 1.0, rel_tol=1e-12
        ), "expanded query not re-normalized"
        assert set(q0) <= set(qm), "alpha must preserve the original query terms"
        added = sorted(set(qm) - set(q0))
        assert "auto" in added, added
        print(f"  '{list(q0)[0]}' -> {sorted(qm)}  (added {added})")
    finally:
        settings.prf_n_relevant, settings.prf_n_nonrelevant = saved


def test_rocchio_respects_the_expansion_cap():
    """Slide 20: long modified queries are expensive, so cap the expansion —
    but never at the cost of an original query term."""
    load_toy()
    saved = (settings.prf_n_relevant, settings.prf_n_nonrelevant,
             settings.prf_expansion_terms)
    try:
        settings.prf_n_relevant, settings.prf_n_nonrelevant = 2, 1
        settings.prf_expansion_terms = 2

        q0 = vsm.query_vector("best car")
        qm = prf.expand(q0, vsm.rank(q0, 3, "exact")["hits"])

        assert set(q0) <= set(qm), "an original term was capped away"
        assert len(qm) <= settings.prf_expansion_terms + len(q0), sorted(qm)
        print(f"  cap={settings.prf_expansion_terms}: "
              f"{len(q0)} original + {len(set(qm) - set(q0))} added = {len(qm)} terms")
    finally:
        (settings.prf_n_relevant, settings.prf_n_nonrelevant,
         settings.prf_expansion_terms) = saved


def test_prf_search_reports_its_expansion():
    load_toy()
    saved = (settings.prf_n_relevant, settings.prf_n_nonrelevant)
    try:
        settings.prf_n_relevant, settings.prf_n_nonrelevant = 1, 1
        plain = vsm.search("insurance", k=5, mode="exact")
        with_prf = vsm.search("insurance", k=5, mode="exact", prf=True)

        assert "expansion" not in plain
        assert with_prf["expansion"], with_prf
        # two retrievals, so more chunks are visited than a single pass
        assert with_prf["scored"] > plain["scored"], (with_prf, plain)
        print(f"  plain scored={plain['scored']}, prf scored={with_prf['scored']}, "
              f"expansion={with_prf['expansion']}")
    finally:
        settings.prf_n_relevant, settings.prf_n_nonrelevant = saved


if __name__ == "__main__":
    tests = [
        test_bm25_against_hand_computation,
        test_bm25_and_vsm_disagree_on_length_penalty,
        test_k3_only_matters_for_repeated_query_terms,
        test_rocchio_expands_and_stays_valid,
        test_rocchio_respects_the_expansion_cap,
        test_prf_search_reports_its_expansion,
    ]
    for fn in tests:
        print(fn.__name__)
        fn()
    init_db()
    index.rebuild_from_db()
    print(f"\nall phase-3 checks passed (index rebuilt: {index.n_chunks} chunks)")
