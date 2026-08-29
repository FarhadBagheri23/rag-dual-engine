"""The inverted index, built from scratch — slides 2-Boolean IR and 8-Scoring.

**The retrieval unit is the document.** Chunks are an artefact of the embedding
model's context window (spec §2.1), not a thing a user searches for, and the
spec is consistent about this: VSM and BM25 "display a ranked list of relevant
*documents*" (§3.3.2), use "TF-IDF weighting for queries and *documents*"
(§3.2.1), eliminate by "only scoring *documents* that contain high-IDF query
terms", and assume "the top k *documents* are relevant" for PRF.

So the index carries two levels, built in one pass:

    chunk level — what §2.1 specifies indexing, and what RAG and snippets need
        postings    term     -> {chunk_id: tf}    the inverted index proper
        forward     chunk_id -> {term: tf}        O(terms) deletion
        chunk_norm  chunk_id -> L2 norm           picking the snippet chunk

    document level — the scoring unit for VSM, BM25 and Rocchio
        doc_forward  doc_id -> {term: tf}    summed over the document's chunks
        doc_postings term   -> {doc_id}      df is *document* frequency
        doc_len      doc_id -> token count   BM25's dl, against a real avgdl
        doc_norm     doc_id -> L2 norm       the `c` in lnc, per document

Scoring documents rather than chunks matters beyond tidiness. `df` means
document frequency, so idf over chunks is not idf. And BM25's `b` exists to
stop long documents winning on term count alone — but chunking deliberately
makes every chunk roughly one length, so normalizing by chunk length corrects
for variance that chunking already removed.

The forward index doubles memory. It buys exact, cheap removal: deleting a
document touches only the terms it actually contained, instead of scanning
every postings list in the vocabulary.
    ponytail: at corpus sizes where that matters, postings move to disk and
    deletion becomes a tombstone + periodic merge. Not at this scale.
"""

import math
import pickle
from collections import defaultdict

from app.core.config import settings
from app.engines.lexical.text import tokenize

INDEX_PATH = settings.index_dir / "inverted.pkl"

# Bump whenever text.py changes how a word becomes a term. The pickle records
# the value it was built under, so a normalization change forces a rebuild the
# same way a new embedding model does in vectordb.stale(). A count-based check
# cannot serve here: changing the analyzer leaves the chunk count identical and
# only the *terms* move, so the index would keep answering queries tokenized one
# way against postings built the other.
ANALYZER = "s11-periods+hyphens/s22-deaccent/s14-fold/s17-porter"

# Bump when the *shape* of the pickle changes, as opposed to how text becomes a
# term. v2 added the document-level maps the engines now score over; a v1 pickle
# has the postings but none of the aggregates, so it must be rebuilt.
SCHEMA = 2


class InvertedIndex:
    def __init__(self):
        self.analyzer = ANALYZER
        self.schema = SCHEMA
        # --- chunk level
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.forward: dict[str, dict[str, int]] = {}
        self.chunk_len: dict[str, int] = {}
        self.chunk_norm: dict[str, float] = {}
        self.chunk_doc: dict[str, str] = {}  # chunk_id -> parent document id
        self.doc_chunks: dict[str, set[str]] = defaultdict(set)  # the inverse
        # --- document level: the scoring unit
        self.doc_forward: dict[str, dict[str, int]] = {}
        self.doc_postings: dict[str, set[str]] = defaultdict(set)
        self.doc_len: dict[str, int] = {}
        self.doc_norm: dict[str, float] = {}
        self._champions: dict[str, list[str]] | None = None  # lazily rebuilt

    # ------------------------------------------------------------------ stats
    @property
    def n_chunks(self) -> int:
        return len(self.forward)

    @property
    def n_docs(self) -> int:
        """N — the collection size every idf is measured against."""
        return len(self.doc_forward)

    @property
    def avg_len(self) -> float:
        """avgdl for BM25, over *documents*.

        Document lengths are what `b` is meant to normalize. Averaging chunk
        lengths instead would hand BM25 a near-constant, because chunking makes
        every chunk about one size by construction.
        """
        return (sum(self.doc_len.values()) / len(self.doc_len)) if self.doc_len else 0.0

    def stale(self) -> bool:
        """True when this index was not built the way the current code builds one.

        Missing counts as stale in both fields: an index pickled before either
        existed was built without stemming, or without the document aggregates,
        and in each case its contents are not what the engines now expect.
        """
        return self.analyzer != ANALYZER or self.schema != SCHEMA

    def df(self, term: str) -> int:
        """Document frequency — the number of *documents* containing the term."""
        return len(self.doc_postings.get(term, ()))

    def chunk_df(self, term: str) -> int:
        """The chunk-level count. Not used for scoring; kept because deletion
        and the champion lists reason about postings, which are chunk-keyed."""
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        """log10(N/df) — the `t` column of slide 7-Scoring s40.

        N is the number of documents and df the number of documents the term
        occurs in, which is what "inverse *document* frequency" means and what
        spec §3.2.1 asks for ("TF-IDF weighting for queries and documents").
        Counting chunks here would make a term's apparent rarity depend on how
        the corpus happened to be split, which is an embedding-window detail.

        Zero for unseen terms, and zero for a term in every document — which is
        also what makes such a term contribute nothing to any score.
        """
        d = self.df(term)
        return math.log10(self.n_docs / d) if d else 0.0

    def eliminate(self, terms: dict) -> dict:
        """Index elimination: keep only the high-idf query terms — s24.

        "catcher in the rye" scores from catcher and rye only. The floor is a
        fraction of the highest idf *in this query*, so it adapts to the query
        rather than to a corpus-wide constant, and a query of uniformly common
        terms keeps all of them instead of eliminating itself to nothing.

        Shared by both lexical engines, and shape-preserving: it filters keys
        and returns the same mapping type it was given, so VSM can pass term
        weights and BM25 can pass query term frequencies.
        """
        if len(terms) < 2:  # nothing to be relatively rare against
            return terms
        floor = settings.elimination_ratio * max(self.idf(t) for t in terms)
        return {t: v for t, v in terms.items() if self.idf(t) >= floor}

    # ----------------------------------------------------------------- writes
    def _recompute_doc(self, doc_id: str):
        """Re-derive one document's aggregate from the chunks it currently has.

        The only place document-level state is written. `add` and `remove` both
        end here, so a document's tf, length and norm cannot disagree with its
        chunks — the same reason corpus.py is the only writer one level up.
        Dropping to zero chunks removes the document entirely.
        """
        for term in self.doc_forward.pop(doc_id, {}):
            entry = self.doc_postings.get(term)
            if entry is not None:
                entry.discard(doc_id)
                if not entry:  # term no longer occurs in any document
                    del self.doc_postings[term]
        self.doc_len.pop(doc_id, None)
        self.doc_norm.pop(doc_id, None)

        cids = self.doc_chunks.get(doc_id) or set()
        if not cids:
            self.doc_chunks.pop(doc_id, None)
            return

        # A document's tf for a term is its total across the document — the
        # chunk boundaries were never part of the retrieval model.
        tf: dict[str, int] = {}
        for cid in cids:
            for term, count in self.forward[cid].items():
                tf[term] = tf.get(term, 0) + count

        self.doc_forward[doc_id] = tf
        self.doc_len[doc_id] = sum(self.chunk_len[cid] for cid in cids)
        # lnc: log tf, no idf, cosine-normalized (slide 7-Scoring s41, s43)
        self.doc_norm[doc_id] = (
            math.sqrt(sum(log_tf(c) ** 2 for c in tf.values())) or 1.0
        )
        for term in tf:
            self.doc_postings[term].add(doc_id)

    def add(self, chunks: list[dict], doc_id: str) -> int:
        """Index one document's chunks. Returns postings added."""
        added = 0
        for chunk in chunks:
            cid = chunk["id"]
            tokens = tokenize(chunk["text"])
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            self.forward[cid] = tf
            self.chunk_len[cid] = len(tokens)
            self.chunk_doc[cid] = doc_id
            self.doc_chunks[doc_id].add(cid)
            self.chunk_norm[cid] = (
                math.sqrt(sum(log_tf(c) ** 2 for c in tf.values())) or 1.0
            )
            for term, count in tf.items():
                self.postings[term][cid] = count
                added += 1

        self._recompute_doc(doc_id)
        self._champions = None
        return added

    def remove(self, chunk_ids: list[str]) -> int:
        """Purge these chunks entirely. Returns postings removed."""
        removed = 0
        touched: set[str] = set()
        for cid in chunk_ids:
            doc_id = self.chunk_doc.pop(cid, None)
            if doc_id is not None:
                touched.add(doc_id)
                self.doc_chunks[doc_id].discard(cid)
            for term in self.forward.pop(cid, {}):
                self.postings[term].pop(cid, None)
                removed += 1
                if not self.postings[term]:  # term no longer occurs anywhere
                    del self.postings[term]
            self.chunk_len.pop(cid, None)
            self.chunk_norm.pop(cid, None)

        # Recompute rather than subtract: removing part of a document changes
        # its norm, and a norm cannot be un-added term by term.
        for doc_id in touched:
            self._recompute_doc(doc_id)

        self._champions = None
        return removed

    def best_chunk(self, doc_id: str, terms) -> str | None:
        """The chunk of this document that matches the query terms best.

        The ranked unit is the document, but §3.3.2 also wants a snippet, and a
        snippet has to come from somewhere specific. Scored as ltc against the
        chunk, so the passage shown is the one that earned the document its
        place rather than merely its first paragraph.
        """
        best, best_score = None, -1.0
        for cid in self.doc_chunks.get(doc_id) or ():
            forward = self.forward.get(cid, {})
            score = sum(
                log_tf(forward[t]) * self.idf(t) for t in terms if t in forward
            )
            if score > best_score:
                best, best_score = cid, score
        return best

    # ------------------------------------------------------- champion lists
    @property
    def champions(self) -> dict[str, list[str]]:
        """Per term, the r documents of highest tf — slide 8-Scoring s26.

        Documents, because they are what gets scored: a champion list of chunks
        would prune the wrong population, and could drop a document whose term
        occurrences are spread thinly across many chunks but which is, in total,
        one of the highest-tf documents for that term.

        Rebuilt lazily after any mutation: r is fixed at build time, so the
        lists go stale the moment the postings change.
        """
        if self._champions is None:
            r = settings.champion_r
            self._champions = {
                term: sorted(
                    docs, key=lambda d: self.doc_forward[d].get(term, 0), reverse=True
                )[:r]
                for term, docs in self.doc_postings.items()
            }
        return self._champions

    def candidates(self, terms, mode: str) -> set[str]:
        """Set A of contending *documents* — slide 8-Scoring s18.

        Shared by VSM and BM25, and by Rocchio through them.
        """
        source = self.champions if mode == "champion" else self.doc_postings
        return {doc_id for t in terms for doc_id in source.get(t, ())}

    # ------------------------------------------------------------ persistence
    def save(self):
        # Only the chunk level and the parentage are stored. Every document
        # aggregate is derived from those two, so persisting them would be a
        # second copy that can disagree with the first.
        with open(INDEX_PATH, "wb") as fh:
            pickle.dump(
                {
                    "postings": dict(self.postings),
                    "forward": self.forward,
                    "chunk_len": self.chunk_len,
                    "chunk_norm": self.chunk_norm,
                    "chunk_doc": self.chunk_doc,
                    "analyzer": self.analyzer,
                    "schema": self.schema,
                },
                fh,
            )

    @classmethod
    def load(cls) -> "InvertedIndex":
        idx = cls()
        if INDEX_PATH.exists():
            with open(INDEX_PATH, "rb") as fh:
                state = pickle.load(fh)
            idx.analyzer = state.get("analyzer")  # absent = pre-stemming pickle
            idx.schema = state.get("schema", 1)  # absent = pre-document pickle
            if idx.stale():
                return idx  # main.py rebuilds from SQLite; don't load v1 shapes
            idx.postings = defaultdict(dict, state["postings"])
            idx.forward = state["forward"]
            idx.chunk_len = state["chunk_len"]
            idx.chunk_norm = state["chunk_norm"]
            idx.chunk_doc = state["chunk_doc"]
            for cid, doc_id in idx.chunk_doc.items():
                idx.doc_chunks[doc_id].add(cid)
            for doc_id in list(idx.doc_chunks):
                idx._recompute_doc(doc_id)
        return idx

    def rebuild_from_db(self):
        """Re-derive the whole index from SQLite, the source of truth."""
        from app.database import documents as db

        self.__init__()
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for chunk in db.all_chunks():
            by_doc[chunk["doc_id"]].append(chunk)
        for doc_id, chunks in by_doc.items():
            self.add(chunks, doc_id)
        self.save()
        return self.n_chunks


def log_tf(tf: int) -> float:
    """1 + log10(tf), the l component — slide 7-Scoring s17."""
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


index = InvertedIndex.load()
