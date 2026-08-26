"""Phase 2 checks: lnc.ltc scoring against hand-computed values, the two
inexact top-K modes, and postings removal.

Run with:  cd backend && python -m tests.test_lexical
"""

import math

from app.core.config import settings
from app.database.session import init_db
from app.engines.lexical import vsm
from app.engines.lexical.heap import TopKMinHeap
from app.engines.lexical.index import index, log_tf

# Toy corpus small enough to score by hand.
TOY = [
    {"id": "c1", "ordinal": 0, "text": "car insurance auto insurance"},
    {"id": "c2", "ordinal": 1, "text": "auto insurance"},
    {"id": "c3", "ordinal": 2, "text": "best car"},
]


def load_toy():
    """Reset the in-memory index to the toy corpus. Never saved to disk."""
    index.__init__()
    index.add(TOY, "toydoc")


def test_heap_selects_top_k():
    heap = TopKMinHeap(3)
    for score, name in [(0.1, "a"), (0.9, "b"), (0.5, "c"), (0.7, "d"), (0.2, "e")]:
        heap.push(score, name)
    assert [n for _, n in heap.ranked()] == ["b", "d", "c"], heap.ranked()
    assert len(heap.heap) == 3
    print("  5 pushed, kept the 3 best in order")


def test_idf_matches_definition():
    load_toy()
    assert index.n_chunks == 3
    # df: car 2, insurance 2, auto 2, best 1
    assert index.df("car") == 2 and index.df("best") == 1
    assert math.isclose(index.idf("car"), math.log10(3 / 2), rel_tol=1e-12)
    assert math.isclose(index.idf("best"), math.log10(3 / 1), rel_tol=1e-12)
    assert index.idf("unseen") == 0.0
    assert math.isclose(log_tf(2), 1 + math.log10(2), rel_tol=1e-12)
    print(f"  idf(car)={index.idf('car'):.6f}  idf(best)={index.idf('best'):.6f}")


def test_lnc_ltc_against_hand_computation():
    """Query 'best car' scored from the slide formulas, independently.

    Nothing below imports project code: the expected numbers come from
    log10 and sqrt applied to the corpus by hand, so agreement means the
    engine implements lnc.ltc and not merely something self-consistent.

        c1 "car insurance auto insurance"   c2 "auto insurance"   c3 "best car"
        N = 3,  df(car) = df(insurance) = df(auto) = 2,  df(best) = 1
    """
    load_toy()

    lg = math.log10
    idf_best, idf_car = lg(3 / 1), lg(3 / 2)

    # query ltc: l = 1+log10(1) = 1 for both terms, times idf, cosine normalized
    q_norm = math.sqrt(idf_best**2 + idf_car**2)
    q_best, q_car = idf_best / q_norm, idf_car / q_norm

    # c3 lnc: best tf=1 -> 1.0, car tf=1 -> 1.0, cosine normalized
    c3_norm = math.sqrt(1.0**2 + 1.0**2)
    expect_c3 = q_best * (1.0 / c3_norm) + q_car * (1.0 / c3_norm)

    # c1 lnc: car 1.0, insurance 1+log10(2), auto 1.0 — 'best' absent
    c1_norm = math.sqrt(1.0**2 + (1 + lg(2)) ** 2 + 1.0**2)
    expect_c1 = q_car * (1.0 / c1_norm)

    result = vsm.search("best car", k=10, mode="exact")
    ranking = [(h["chunk_id"], h["score"]) for h in result["hits"]]

    assert [cid for cid, _ in ranking] == ["c3", "c1"], ranking
    assert math.isclose(ranking[0][1], expect_c3, abs_tol=1e-6), (ranking[0], expect_c3)
    assert math.isclose(ranking[1][1], expect_c1, abs_tol=1e-6), (ranking[1], expect_c1)

    # c2 shares no query term, so it must not be scored at all
    assert "c2" not in [cid for cid, _ in ranking]

    # and the query vector really is unit length (the 'c' in ltc)
    qv = vsm.query_vector("best car")
    assert math.isclose(math.sqrt(sum(w * w for w in qv.values())), 1.0, rel_tol=1e-12)
    print(f"  c3={ranking[0][1]:.7f} (expected {expect_c3:.7f})")
    print(f"  c1={ranking[1][1]:.7f} (expected {expect_c1:.7f})")


def test_champion_lists_are_top_r_by_tf_and_non_safe():
    """Champion lists are built on raw tf, but ranking is cosine-normalized —
    so a small r can miss the true top hit. That is the documented trade-off
    of inexact retrieval (slide 8-Scoring s16, safe vs. non-safe ranking),
    and this corpus exhibits it.

        c1 "car insurance auto insurance"  insurance tf=2, but 3 distinct terms
        c2 "auto insurance"                insurance tf=1, only 2 terms
        champion by tf   -> c1
        cosine ranking   -> c2 first, because c1 pays a longer-vector penalty
    """
    load_toy()
    original_r = settings.champion_r
    try:
        settings.champion_r = 1
        index._champions = None
        assert index.champions["insurance"] == ["c1"], index.champions["insurance"]

        champion = vsm.search("insurance", k=10, mode="champion")
        exact = vsm.search("insurance", k=10, mode="exact")

        assert champion["scored"] == 1 and exact["scored"] == 2
        assert [h["chunk_id"] for h in champion["hits"]] == ["c1"]
        assert exact["hits"][0]["chunk_id"] == "c2", exact["hits"]
        print(f"  r=1: scored {champion['scored']} vs {exact['scored']}; "
              f"champion top={champion['hits'][0]['chunk_id']} "
              f"exact top={exact['hits'][0]['chunk_id']} (non-safe, as expected)")

        # widen r until the contender set covers both -> agreement is restored
        settings.champion_r = 2
        index._champions = None
        champion = vsm.search("insurance", k=10, mode="champion")
        assert [h["chunk_id"] for h in champion["hits"]] == \
               [h["chunk_id"] for h in exact["hits"]], champion["hits"]
        print("  r=2: champion ranking matches exact ranking")
    finally:
        settings.champion_r = original_r
        index._champions = None


def test_index_elimination_drops_low_idf_terms():
    load_toy()
    # 'car' (idf .176) vs 'best' (idf .477): ratio .3 -> floor .143, both kept.
    # Raise the ratio so only the rarest term survives.
    original = settings.elimination_ratio
    try:
        settings.elimination_ratio = 0.9
        kept = vsm._eliminate(vsm.query_vector("best car"))
        assert set(kept) == {"best"}, kept

        settings.elimination_ratio = 0.0
        kept = vsm._eliminate(vsm.query_vector("best car"))
        assert set(kept) == {"best", "car"}, kept

        # a query whose terms all share one idf must not eliminate itself away
        settings.elimination_ratio = 0.9
        kept = vsm._eliminate(vsm.query_vector("car auto"))
        assert set(kept) == {"car", "auto"}, kept
        print("  high floor keeps only 'best'; uniform-idf query keeps all terms")
    finally:
        settings.elimination_ratio = original


def test_remove_purges_every_posting():
    """Sync foundation: after removal the vocabulary is empty, not merely
    unreachable — a term with no postings must not linger in the dictionary."""
    load_toy()
    n_postings = sum(len(p) for p in index.postings.values())
    assert n_postings == 7, n_postings  # c1:3 + c2:2 + c3:2 distinct terms

    removed = index.remove(["c1", "c2", "c3"])

    assert removed == 7, removed
    assert index.postings == {}, index.postings
    assert index.forward == {} and index.norm == {} and index.doc_len == {}
    assert index.n_chunks == 0
    assert vsm.search("car", mode="exact")["hits"] == []
    print(f"  removed all {removed} postings, vocabulary empty")


if __name__ == "__main__":
    tests = [
        test_heap_selects_top_k,
        test_idf_matches_definition,
        test_lnc_ltc_against_hand_computation,
        test_champion_lists_are_top_r_by_tf_and_non_safe,
        test_index_elimination_drops_low_idf_terms,
        test_remove_purges_every_posting,
    ]
    for fn in tests:
        print(fn.__name__)
        fn()
    # restore the real index the toy corpus displaced, re-deriving it from
    # SQLite — which is exactly the recovery path main.py uses at startup
    init_db()
    index.rebuild_from_db()
    print(f"\nall phase-2 checks passed (index rebuilt: {index.n_chunks} chunks)")
