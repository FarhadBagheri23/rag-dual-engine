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
from app.engines.lexical.text import tokenize

# Toy corpus small enough to score by hand.
TOY = [
    {"id": "c1", "ordinal": 0, "text": "car insurance auto insurance"},
    {"id": "c2", "ordinal": 1, "text": "auto insurance"},
    {"id": "c3", "ordinal": 2, "text": "best car"},
]


def load_toy():
    """Reset the in-memory index to the toy corpus. Never saved to disk.

    One chunk per document, with the document id equal to the chunk id. The
    slides treat each of these strings as a *document* — which is the unit the
    engines score — so every hand-computed number below is the slide's own.
    """
    index.__init__()
    for chunk in TOY:
        index.add([chunk], chunk["id"])


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
    # N is the *document* count — the collection size idf divides by.
    assert index.n_docs == 3
    # df: car 2, insurance 2, auto 2, best 1 — documents, not chunks
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
    ranking = [(h["doc_id"], h["score"]) for h in result["hits"]]

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


def test_matches_slide_43_worked_example():
    """Slide 7-Scoring s43 computes lnc.ltc end to end, and this reproduces it.

        Document: car insurance auto insurance
        Query:    best car insurance
        df: auto 5000, best 50000, car 10000, insurance 1000   (N = 1,000,000)
        Doc length = 1.92        Score = 0.8

    The million-document collection behind those df values is not something we
    can index, so the *document* is indexed for real and only the collection
    statistics are substituted. Everything downstream — the query vector, both
    cosine normalizations, the scoring loop — is the shipping code.
    """
    index.__init__()
    index.add([{"id": "d1", "ordinal": 0, "text": "car insurance auto insurance"}], "s43")

    n_docs = 1_000_000
    # keyed by term, not by the surface word: 'insurance' indexes as 'insur'
    slide_df = {
        tokenize(word)[0]: df
        for word, df in [("auto", 5000), ("best", 50000), ("car", 10000), ("insurance", 1000)]
    }
    index.df = lambda t: slide_df.get(t, 0)
    index.idf = lambda t: math.log10(n_docs / slide_df[t]) if t in slide_df else 0.0
    try:
        query = vsm.query_vector("best car insurance")
        assert {t: round(w, 2) for t, w in query.items()} == {
            tokenize("best")[0]: 0.34,
            tokenize("car")[0]: 0.52,
            tokenize("insurance")[0]: 0.78,
        }, query

        # The document's norm, keyed by document id — s43's "Doc length 1.92".
        assert round(index.doc_norm["s43"], 2) == 1.92, index.doc_norm["s43"]

        score = vsm.rank(query, k=10, mode="exact")["hits"][0]["score"]
        assert math.isclose(score, 0.8, abs_tol=0.005), score
        print(f"  doc length 1.92, score {score:.4f} — slide 43 says 1.92 and 0.8")
    finally:
        del index.df, index.idf  # drop the stubs, restoring the real methods


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
        # Ask the analyzer for the term rather than hardcoding the surface form:
        # postings are keyed by the *term* ("insur" once Porter has run), and a
        # test that spells it out breaks on every normalization change.
        term = tokenize("insurance")[0]
        assert index.champions[term] == ["c1"], index.champions[term]

        champion = vsm.search("insurance", k=10, mode="champion")
        exact = vsm.search("insurance", k=10, mode="exact")

        assert champion["scored"] == 1 and exact["scored"] == 2
        assert [h["doc_id"] for h in champion["hits"]] == ["c1"]
        assert exact["hits"][0]["doc_id"] == "c2", exact["hits"]
        print(f"  r=1: scored {champion['scored']} vs {exact['scored']}; "
              f"champion top={champion['hits'][0]['doc_id']} "
              f"exact top={exact['hits'][0]['doc_id']} (non-safe, as expected)")

        # widen r until the contender set covers both -> agreement is restored
        settings.champion_r = 2
        index._champions = None
        champion = vsm.search("insurance", k=10, mode="champion")
        assert [h["doc_id"] for h in champion["hits"]] == \
               [h["doc_id"] for h in exact["hits"]], champion["hits"]
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
        kept = index.eliminate(vsm.query_vector("best car"))
        assert set(kept) == {"best"}, kept

        settings.elimination_ratio = 0.0
        kept = index.eliminate(vsm.query_vector("best car"))
        assert set(kept) == {"best", "car"}, kept

        # a query whose terms all share one idf must not eliminate itself away
        settings.elimination_ratio = 0.9
        kept = index.eliminate(vsm.query_vector("car auto"))
        assert set(kept) == {"car", "auto"}, kept
        print("  high floor keeps only 'best'; uniform-idf query keeps all terms")
    finally:
        settings.elimination_ratio = original


def test_a_document_is_ranked_once_and_scored_whole():
    """The bug this file exists to prevent: chunks competing as separate hits.

    One document split into three chunks, against one document that is a single
    chunk. If chunks were the retrieval unit, `split` would occupy three of the
    ranked rows and its term occurrences would be divided across them.

        split  = "car" + "car insurance" + "insurance auto"   (car tf=2 total)
        single = "car auto"
        other  = "best rank"        a third document, so 'car' is discriminating

    Scored as documents: two rows, and split's tf for 'car' is 2 — a number
    that exists only once the chunk boundaries are ignored.
    """
    index.__init__()
    index.add(
        [
            {"id": "s1", "ordinal": 0, "text": "car"},
            {"id": "s2", "ordinal": 1, "text": "car insurance"},
            {"id": "s3", "ordinal": 2, "text": "insurance auto"},
        ],
        "split",
    )
    index.add([{"id": "u1", "ordinal": 0, "text": "car auto"}], "single")
    index.add([{"id": "o1", "ordinal": 0, "text": "best rank"}], "other")

    car = tokenize("car")[0]
    assert index.n_chunks == 5 and index.n_docs == 3
    # df counts documents: 'car' is in two of the three, not in three of the
    # five chunks — the distinction this whole change is about.
    assert index.df(car) == 2, index.df(car)
    assert index.chunk_df(car) == 3, index.chunk_df(car)
    assert math.isclose(index.idf(car), math.log10(3 / 2), rel_tol=1e-12)
    # tf is summed over the document's chunks.
    assert index.doc_forward["split"][car] == 2
    # dl is the whole document, which is what BM25's b normalizes against.
    assert index.doc_len["split"] == 5 and index.doc_len["single"] == 2

    hits = vsm.search("car insurance", k=10, mode="exact")["hits"]
    ids = [h["doc_id"] for h in hits]
    assert ids == sorted(set(ids), key=ids.index), f"a document ranked twice: {ids}"
    assert set(ids) == {"split", "single"}, ids

    # And the snippet still comes from a real chunk — the one that matches best.
    best = index.best_chunk("split", {tokenize("insurance")[0]})
    assert best in {"s2", "s3"}, best

    # A term in *every* document carries no information and must score zero,
    # which is only true when df counts documents.
    assert index.idf(tokenize("auto")[0]) > 0  # 2 of 3 documents
    index.add([{"id": "o2", "ordinal": 1, "text": "auto"}], "other")
    assert index.idf(tokenize("auto")[0]) == 0.0, "term in all 3 docs still scores"
    print(f"  5 chunks -> 2 ranked documents {ids}; car tf=2 across split's chunks")


def test_remove_purges_every_posting():
    """Sync foundation: after removal the vocabulary is empty, not merely
    unreachable — a term with no postings must not linger in the dictionary."""
    load_toy()
    n_postings = sum(len(p) for p in index.postings.values())
    assert n_postings == 7, n_postings  # c1:3 + c2:2 + c3:2 distinct terms

    removed = index.remove(["c1", "c2", "c3"])

    assert removed == 7, removed
    assert index.postings == {}, index.postings
    assert index.forward == {} and index.chunk_norm == {} and index.chunk_len == {}
    # The document aggregates must go too, or idf keeps dividing by a
    # collection size that no longer has any documents in it.
    assert index.doc_forward == {} and index.doc_postings == {}
    assert index.doc_len == {} and index.doc_norm == {}
    assert index.n_chunks == 0 and index.n_docs == 0
    assert vsm.search("car", mode="exact")["hits"] == []
    print(f"  removed all {removed} postings, both levels empty")


if __name__ == "__main__":
    tests = [
        test_heap_selects_top_k,
        test_idf_matches_definition,
        test_lnc_ltc_against_hand_computation,
        test_matches_slide_43_worked_example,
        test_champion_lists_are_top_r_by_tf_and_non_safe,
        test_index_elimination_drops_low_idf_terms,
        test_a_document_is_ranked_once_and_scored_whole,
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
