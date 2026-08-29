"""The score matrix behind the heatmap: does every column add up?

The heatmap's whole claim is that a document's column sums to the score printed
beside it. If the breakdown ever drifts from the scoring loop — someone edits
one and not the other — the picture becomes a confident lie, and nothing else
in the suite would notice, because the *scores* would still be right.

Run with:  cd backend && python -m tests.test_explain
"""

import math

from app.database.session import init_db
from app.engines.lexical import bm25, prf, vsm
from app.engines.lexical.index import index
from tests.test_lexical import load_toy


def test_vsm_columns_sum_to_the_score():
    load_toy()
    result = vsm.search("car insurance", k=10, mode="exact")
    assert result["hits"], result

    for hit in result["hits"]:
        column = result["contrib"][hit["doc_id"]]
        assert math.isclose(sum(column.values()), hit["score"], rel_tol=1e-12), (
            hit["doc_id"],
            column,
            hit["score"],
        )
    print(f"  {len(result['hits'])} columns, each summing to its cosine score")


def test_bm25_columns_sum_to_the_score():
    load_toy()
    result = bm25.search("car insurance", k=10, mode="exact")
    assert result["hits"], result

    for hit in result["hits"]:
        column = result["contrib"][hit["doc_id"]]
        assert math.isclose(sum(column.values()), hit["score"], rel_tol=1e-12), (
            hit["doc_id"],
            column,
            hit["score"],
        )
    print(f"  {len(result['hits'])} columns, each summing to its BM25 score")


def test_rows_are_the_terms_actually_ranked_on():
    """Not the terms the user typed — the ones left after the query vector was
    built. A stopword-ish term with idf 0 never reaches the matrix, and index
    elimination removes more."""
    load_toy()

    # "auto" occurs in c1 and c2 of three documents, so it survives; a word
    # absent from the corpus contributes nothing and must not become a row.
    result = vsm.search("auto zzzznotacorpusword", k=10, mode="exact")
    assert set(result["weights"]) == {"auto"}, result["weights"]

    # Whatever elimination decides to keep — that decision is checked in
    # test_lexical — the matrix must follow it. No column may hold a term the
    # engine did not rank on, or the columns would stop summing to the scores.
    eliminated = vsm.search("best car", k=10, mode="elimination")
    kept = set(eliminated["weights"])
    assert kept, eliminated
    for column in eliminated["contrib"].values():
        assert set(column) <= kept, (column, kept)
    print(f"  rows follow the query vector: idf-0 dropped, {sorted(kept)} kept")


def test_a_term_matching_nothing_is_absent_from_every_column():
    """The empty row the UI draws. "best" is in exactly one document, so the
    other two must have no cell for it at all — not a zero, an absence."""
    load_toy()
    result = vsm.search("best insurance", k=10, mode="exact")

    holders = {d for d, col in result["contrib"].items() if "best" in col}
    assert holders == {"c3"}, result["contrib"]
    assert "best" in result["weights"], result["weights"]
    print("  'best' is a row, but a cell only in the one document that has it")


def test_rocchio_reports_the_second_pass():
    """PRF ranks twice and only the second ranking is displayed, so the matrix
    must describe that one — including the terms Rocchio added."""
    load_toy()
    result = vsm.search("insurance", k=10, mode="exact", prf=True)
    assert result["hits"], result

    for hit in result["hits"]:
        column = result["contrib"][hit["doc_id"]]
        assert math.isclose(sum(column.values()), hit["score"], rel_tol=1e-12), (
            hit["doc_id"],
            column,
        )

    # The expansion terms are part of the query that produced these scores, so
    # they have to be available as rows.
    assert set(result["expansion"]) <= set(result["weights"]), (
        result["expansion"],
        result["weights"],
    )
    print(f"  second pass reported, {len(result['expansion'])} expansion terms as rows")


if __name__ == "__main__":
    tests = [
        test_vsm_columns_sum_to_the_score,
        test_bm25_columns_sum_to_the_score,
        test_rows_are_the_terms_actually_ranked_on,
        test_a_term_matching_nothing_is_absent_from_every_column,
        test_rocchio_reports_the_second_pass,
    ]
    for fn in tests:
        print(fn.__name__)
        fn()
    init_db()
    index.rebuild_from_db()
    print(f"\nall score-matrix checks passed (index rebuilt: {index.n_chunks} chunks)")
