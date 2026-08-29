"""Phase 1 check: parsing produces text, chunking overlaps, deletion is total.

Run with:  cd backend && python -m tests.test_ingest
"""

from pathlib import Path

from app.core.config import settings
from app.database import documents as db
from app.database.session import init_db
from app.services import corpus, ingest

CORPUS = Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


def test_parse_both_formats():
    for path in sorted(CORPUS.iterdir()):
        text, title = ingest.parse(path, path.name)
        assert len(text.split()) > 100, f"{path.name}: only {len(text.split())} words"
        assert title, f"{path.name}: no title"
    print(f"  parsed {len(list(CORPUS.iterdir()))} files (.pdf and .docx)")


def test_title_never_comes_from_the_temp_file():
    """Uploads are parsed from a temp path, so a title fallback that reads the
    path produces names like `tmpoqta_c1i`. Precedence is: what the uploader
    typed, then the document's own metadata title, then the real filename."""
    init_db()
    src = CORPUS / "tf-idf.pdf"
    # a temp-style name, exactly as tempfile.NamedTemporaryFile would produce
    tmp = Path(settings.upload_dir) / "tmpq7x2ab9z.pdf"

    def upload(filename, title=None):
        tmp.write_bytes(src.read_bytes())
        doc = corpus.add(tmp, filename, title)
        corpus.remove(doc["id"])
        return doc["title"]

    assert upload("Learning to Rank_new.pdf", "Learning to Rank") == "Learning to Rank"
    # tf-idf.pdf carries its own metadata title, which beats the filename
    assert upload("Learning to Rank_new.pdf") == "TF-IDF Weighting"

    # and with no metadata title, the filename stem — never the temp stem
    original = ingest.PARSERS[".pdf"]
    ingest.PARSERS[".pdf"] = lambda p: (original(p)[0], "")
    try:
        assert upload("Learning to Rank_new.pdf") == "Learning to Rank_new"
    finally:
        ingest.PARSERS[".pdf"] = original
    print("  typed title > metadata title > filename stem; temp name never used")


def test_chunk_overlap():
    """Consecutive chunks must share the tail/head words, or context is lost
    at every boundary (RAG slides s32)."""
    words = [f"w{i}" for i in range(1200)]
    chunks = ingest.chunk(" ".join(words), "testdoc")
    assert len(chunks) > 1, "1200 words must split at chunk_size=500"

    for c in chunks:
        assert len(c["text"].split()) <= settings.chunk_size, "chunk over the limit"

    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    shared = set(first[-settings.chunk_overlap:]) & set(second)
    assert shared, "no overlap between consecutive chunks"

    assert [c["ordinal"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["id"] == "testdoc:0"
    print(f"  1200 words -> {len(chunks)} chunks, {len(shared)} words overlapping")


def test_add_then_remove_leaves_nothing():
    """The foundation of the sync rubric: after remove(), no trace survives."""
    init_db()
    src = CORPUS / "tf-idf.pdf"
    tmp = Path(settings.upload_dir) / "_test_copy.pdf"
    tmp.write_bytes(src.read_bytes())

    doc = corpus.add(tmp, "tf-idf.pdf")
    doc_id = doc["id"]
    assert db.get(doc_id) is not None
    assert len(db.chunk_ids(doc_id)) == doc["n_chunks"] > 0
    stored = settings.upload_dir / f"{doc_id}.pdf"
    assert stored.exists(), "original bytes not kept"

    result = corpus.remove(doc_id)

    assert result["chunks_removed"] == doc["n_chunks"]
    assert db.get(doc_id) is None, "document row survived"
    assert db.chunk_ids(doc_id) == [], "chunk rows survived"
    assert not stored.exists(), "stored file survived"
    print(f"  added {doc['n_chunks']} chunks, removed all of them")


if __name__ == "__main__":
    for fn in (
        test_parse_both_formats,
        test_title_never_comes_from_the_temp_file,
        test_chunk_overlap,
        test_add_then_remove_leaves_nothing,
    ):
        print(fn.__name__)
        fn()
    print("\nall phase-1 checks passed")
