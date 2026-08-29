"""Analysis-pipeline checks against slide 3-Document Preprocessing.

Every assertion below is one of the slides' own examples, so a failure here
means the code stopped matching the course material rather than merely
stopped matching a previous run.

Run with:  cd backend && python -m tests.test_normalize
"""

from app.database.session import init_db
from app.engines.lexical import vsm
from app.engines.lexical.index import ANALYZER, index
from app.engines.lexical.text import normalize, stem, tokenize

TOY = [
    {"id": "c1", "ordinal": 0, "text": "The organizing committee organizes a démocratic vote"},
    {"id": "c2", "ordinal": 1, "text": "Anti-discriminatory policy at the U.S.A. office"},
]


def load_toy():
    # One chunk per document, so c1/c2/… are the units the engines rank.
    index.__init__()
    for chunk in TOY:
        index.add([chunk], chunk["id"])


def test_s11_periods_and_hyphens_are_deleted():
    # "deleting periods to form a term: U.S.A., USA -> USA"
    assert tokenize("U.S.A.") == tokenize("USA") == ["usa"]
    # "deleting hyphens to form a term"
    assert tokenize("anti-discriminatory") == tokenize("antidiscriminatory")
    # ...but a sentence boundary the PDF left unspaced must still be two terms,
    # which is why the period rule is restricted to acronym runs.
    assert tokenize("the index.The next") == ["the", "index", "the", "next"]
    print("  U.S.A.->usa, anti-discriminatory->antidiscriminatory, index.The stays two terms")


def test_s22_accents_are_folded():
    # the slide's own pair, plus the French example from the same page
    assert normalize("Tübingen") == "tubingen"
    assert tokenize("résumé") == tokenize("resume")
    # NFKD's compatibility half also unpicks the ligatures PDF text extraction
    # emits, which is why `ﬁle` and `file` are the same term
    assert tokenize("ﬁle") == tokenize("file")
    print("  Tübingen->tubingen, résumé==resume, ﬁle==file")


def test_s14_case_folding():
    assert tokenize("General Motors") == tokenize("general motors")
    print("  case folded")


def test_s17_stemming_builds_equivalence_classes():
    # s15's example: "organize, organizes, and organizing"
    assert stem("organize") == stem("organizes") == stem("organizing") == "organ"
    # s18's rules, applied by the full algorithm
    assert stem("caresses") == "caress"  # sses -> ss
    assert stem("ponies") == "poni"  # ies -> i
    assert stem("replacement") == "replac"  # (m>1) EMENT ->
    assert stem("cement") == "cement"  # ...but not when m is 1
    # s17: "the exact stemmed form does not matter, only the resulting
    # equivalence classes" — `organ` is not a word, and does not need to be.
    print("  organize/organizes/organizing -> organ; s18's rules hold")


def test_non_latin_and_numbers_pass_through():
    """Porter is English-specific; it must not mangle what it cannot analyse."""
    assert tokenize("بازیابی اطلاعات") == ["بازیابی", "اطلاعات"]
    assert tokenize("BM25 in 2024") == ["bm25", "in", "2024"]
    # Arabic harakat are combining marks, so de-accenting strips them — which
    # is what a user typing the bare consonants needs (s22's criterion).
    assert tokenize("مُحَمَّد") == tokenize("محمد")
    print("  Persian/Arabic/CJK and numbers survive; harakat folded")


def test_index_and_query_meet_in_the_same_form():
    """s11: 'Crucial: Need to normalize indexed text as well as query terms
    into the same form.' The proof is a query that shares no surface word with
    the document it must retrieve."""
    load_toy()
    # "organize" never appears in c1 — only "organizing" and "organizes" do.
    hits = vsm.search("organize", k=5, mode="exact")["hits"]
    assert hits and hits[0]["doc_id"] == "c1", hits
    # and the hyphen/period rules reach the index the same way
    assert vsm.search("antidiscriminatory", k=5, mode="exact")["hits"][0]["doc_id"] == "c2"
    assert vsm.search("usa", k=5, mode="exact")["hits"][0]["doc_id"] == "c2"
    print("  'organize' retrieves a doc containing only 'organizing'/'organizes'")


def test_analyzer_change_forces_a_rebuild():
    """The postings are only meaningful under the analyzer that built them, so
    the index records it — the same guard vectordb.stale() applies to the
    embedding model, and for the same reason: the chunk count does not move."""
    load_toy()
    assert not index.stale()
    index.analyzer = "something-else"
    assert index.stale(), "an analyzer change must be detected"
    index.analyzer = ANALYZER
    print(f"  stale() detects an analyzer change ({ANALYZER})")


if __name__ == "__main__":
    tests = [
        test_s11_periods_and_hyphens_are_deleted,
        test_s22_accents_are_folded,
        test_s14_case_folding,
        test_s17_stemming_builds_equivalence_classes,
        test_non_latin_and_numbers_pass_through,
        test_index_and_query_meet_in_the_same_form,
        test_analyzer_change_forces_a_rebuild,
    ]
    for fn in tests:
        print(fn.__name__)
        fn()
    init_db()
    index.rebuild_from_db()
    print(f"\nall preprocessing checks passed (index rebuilt: {index.n_chunks} chunks)")
