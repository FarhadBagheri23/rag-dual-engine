"""Phase 4 checks: the two indexes stay synchronized, and RAG citation
parsing works without needing a provider.

First run downloads the embedding model (~90 MB).

Run with:  cd backend && python -m tests.test_rag_sync
"""

from pathlib import Path

from app.core.config import settings
from app.database import documents as db
from app.database.session import init_db
from app.engines.lexical.index import index
from app.engines.semantic import rag, vectordb
from app.services import corpus

CORPUS = Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


def test_citation_parsing():
    """Pure parsing — no API key, no network."""
    chunks = [
        {"id": "a:0", "title": "Alpha", "text": "alpha text"},
        {"id": "b:0", "title": "Beta", "text": "beta text"},
        {"id": "c:0", "title": "Gamma", "text": "gamma text"},
    ]
    answer = "Ranking uses tf-idf [Doc 1] and length normalization [doc 3]."
    cited = {int(n) for n in rag.CITATION_RE.findall(answer)}
    assert cited == {1, 3}, cited

    citations = [
        {"doc_number": i, "chunk_id": c["id"], "title": c["title"]}
        for i, c in enumerate(chunks, start=1)
        if i in cited
    ]
    assert [c["title"] for c in citations] == ["Alpha", "Gamma"], citations
    print(f"  '{answer[:40]}…' -> cited {sorted(cited)}")


def test_prompt_guards_against_injection():
    """Retrieved text is untrusted (RAG slides s44) — the prompt must fence it
    and say so, or a document in the corpus can issue instructions."""
    context = rag._context([{"id": "x:0", "title": "T", "text": "body"}])
    assert "[Doc 1]" in context and "source: T" in context

    prompt = rag.PROMPT.format(
        context=context, question="q", conversation="", resolved=""
    )
    assert "<passages>" in prompt and "</passages>" in prompt, "context not fenced"
    assert "not instructions" in prompt, "no injection guard"
    assert "ONLY" in prompt and "I don't know" in prompt, "no grounding instruction"

    # Conversation history is fenced separately and marked as non-evidence:
    # it exists to resolve "just it?", not to be quoted back as fact.
    with_history = rag.PROMPT.format(
        context=context,
        question="just it?",
        conversation="\n<conversation>\nUser: earlier\n</conversation>\n",
        resolved="Which in this conversation means: is that the only reason?\n",
    )
    assert "<conversation>" in with_history, "history not fenced"
    assert "not evidence" in with_history, "history not marked as non-evidence"
    print("  context fenced, injection guard and grounding instruction present")
    print("  history fenced separately and marked as non-evidence")


def test_query_rewriting_resolves_follow_ups_and_fails_safe():
    """RAG slides s24. A follow-up carries its meaning in the conversation, so
    "just it?" must be resolved before it is embedded — otherwise it retrieves
    nothing, trips the relevance floor, and the user is told their own
    follow-up is out of corpus. Rewriting is an optimisation, though: when the
    model misbehaves the original question has to survive."""
    from app.engines.semantic import llm

    real = llm.complete
    transcript = "User: Why normalise document length?\nAssistant: Because …"
    try:
        llm.complete = lambda *a, **k: '  "Why is document length normalised?"  '
        assert (
            rag._rewrite("just it?", transcript, None)
            == "Why is document length normalised?"
        ), "quotes and whitespace not stripped"

        # Each failure mode falls back to the question the user actually asked.
        for name, stub in [
            ("empty reply", lambda *a, **k: "   "),
            ("runaway reply", lambda *a, **k: "x" * 501),
            ("provider down", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())),
        ]:
            llm.complete = stub
            assert rag._rewrite("just it?", transcript, None) == "just it?", name
    finally:
        llm.complete = real

    # And with no history there is nothing to resolve, so no call is made.
    assert rag._transcript([]) == ""
    print("  'just it?' -> resolved; empty/runaway/failed rewrites fall back")


def test_provider_errors_map_to_actionable_status():
    """A restricted key or unknown model must not surface as a bare 500 — the
    provider's message names the fix, so it has to reach the caller."""
    from app.core.exceptions import ProviderError

    fixable = ProviderError(403, "Access denied for model 'x'. Verify key.")
    assert fixable.status_code == 400, fixable.status_code
    assert "Verify key" in fixable.detail

    upstream = ProviderError(503, "upstream unavailable")
    assert upstream.status_code == 502, upstream.status_code

    network = ProviderError(None, "connection reset")
    assert network.status_code == 502, network.status_code
    print("  403 -> 400 (caller can fix), 503/None -> 502 (upstream fault)")


def counts():
    """(sqlite, postings, vectors) — the three numbers that must agree."""
    return len(db.all_chunks()), index.n_chunks, vectordb.count()


def test_both_indexes_stay_synchronized():
    """The rubric line: add and remove must leave SQLite, the inverted index
    and the vector store agreeing at every step."""
    init_db()

    # Reconcile first, exactly as main.py does at startup — a corpus indexed
    # before the vector store existed starts out with zero vectors.
    n_chunks = len(db.all_chunks())
    if index.n_chunks != n_chunks:
        index.rebuild_from_db()
    if vectordb.count() != n_chunks:
        print(f"    reconciling vector store: {vectordb.count()} -> {n_chunks}")
        vectordb.rebuild_from_db()

    before = counts()
    assert len(set(before)) == 1, f"stores disagreed after reconcile: {before}"

    src = CORPUS / "vector-space-model.pdf"
    tmp = settings.upload_dir / "_sync_test.pdf"
    tmp.write_bytes(src.read_bytes())
    doc = corpus.add(tmp, "vector-space-model.pdf")

    after_add = counts()
    assert len(set(after_add)) == 1, f"stores disagreed after add: {after_add}"
    assert after_add[0] == before[0] + doc["n_chunks"], (before, after_add)

    # the new chunks are reachable from both engines
    semantic = vectordb.query("how are documents represented as vectors", k=3)
    assert any(h["chunk_id"].startswith(doc["id"]) for h in semantic), semantic
    assert index.candidates(["vector"], "exact"), "not in the inverted index"

    result = corpus.remove(doc["id"])

    after_remove = counts()
    assert len(set(after_remove)) == 1, f"stores disagreed after remove: {after_remove}"
    assert after_remove == before, (before, after_remove)
    assert result["chunks_removed"] == doc["n_chunks"]
    assert result["postings_removed"] > 0 and result["vectors_removed"] > 0, result

    # and gone from the dense side too, not merely from SQLite
    gone = vectordb.query("how are documents represented as vectors", k=5)
    assert not any(h["chunk_id"].startswith(doc["id"]) for h in gone), gone

    print(f"  sqlite/postings/vectors: {before} -> {after_add} -> {after_remove}")
    print(f"  removed {result['chunks_removed']} chunks, "
          f"{result['postings_removed']} postings, {result['vectors_removed']} vectors")


def test_failed_embedding_rolls_the_whole_document_back():
    """Embedding runs inside the upload request and can fail — no key, no
    model, disk full. Before the rollback, a raising vectordb.add left the
    document in SQLite and the inverted index but not Chroma: live in VSM and
    BM25, invisible to RAG, while the caller got a 500 and assumed nothing had
    happened. The stores must end where they started."""
    init_db()
    before = counts()

    src = CORPUS / "okapi-bm25.pdf"
    tmp = settings.upload_dir / "_rollback_test.pdf"
    tmp.write_bytes(src.read_bytes())

    real_add = vectordb.add
    vectordb.add = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chroma down"))
    try:
        corpus.add(tmp, "okapi-bm25.pdf")
        raise AssertionError("corpus.add swallowed the failure")
    except RuntimeError:
        pass  # the caller sees the real error, which is the point
    finally:
        vectordb.add = real_add
        tmp.unlink(missing_ok=True)

    after = counts()
    assert after == before, f"drift after a failed add: {before} -> {after}"
    print(f"  vectordb.add raised -> {after}, unchanged from {before}")


def test_semantic_beats_keyword_on_vocabulary_mismatch():
    """The reason the dense index exists: a query sharing no words with the
    answer. If this ever fails, the embedding half is not earning its keep."""
    init_db()
    hits = vectordb.query("what stops a language model from making things up", k=3)
    assert hits, "vector store is empty — index the sample corpus first"
    top = db.all_chunks()
    titles = {c["id"]: c["title"] for c in top}
    ranked = [titles.get(h["chunk_id"], "?") for h in hits]
    assert "Retrieval-Augmented Generation" in ranked[:2], ranked
    print(f"  no shared keywords -> {ranked[:2]}")


if __name__ == "__main__":
    for fn in [
        test_citation_parsing,
        test_prompt_guards_against_injection,
        test_query_rewriting_resolves_follow_ups_and_fails_safe,
        test_provider_errors_map_to_actionable_status,
        test_both_indexes_stay_synchronized,
        test_failed_embedding_rolls_the_whole_document_back,
        test_semantic_beats_keyword_on_vocabulary_mismatch,
    ]:
        print(fn.__name__)
        fn()
    print("\nall phase-4 checks passed")
