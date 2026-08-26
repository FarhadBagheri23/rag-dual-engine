"""The inverted index, built from scratch — slides 2-Boolean IR and 8-Scoring.

Structure
    postings   term     -> {chunk_id: tf}     the inverted index proper
    forward    chunk_id -> {term: tf}         needed for O(terms) deletion
    champions  term     -> [chunk_id, ...]    top-r by tf, slide 8-Scoring s26

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


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.forward: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}  # chunk_id -> token count (BM25, phase 3)
        self.norm: dict[str, float] = {}  # chunk_id -> L2 norm of its lnc vector
        self.chunk_doc: dict[str, str] = {}  # chunk_id -> parent document id
        self._champions: dict[str, list[str]] | None = None  # lazily rebuilt

    # ------------------------------------------------------------------ stats
    @property
    def n_chunks(self) -> int:
        return len(self.forward)

    @property
    def avg_len(self) -> float:
        return (sum(self.doc_len.values()) / len(self.doc_len)) if self.doc_len else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        """log10(N/df) — slide 7-Scoring s22. Zero for unseen terms."""
        d = self.df(term)
        return math.log10(self.n_chunks / d) if d else 0.0

    # ----------------------------------------------------------------- writes
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
            self.doc_len[cid] = len(tokens)
            self.chunk_doc[cid] = doc_id
            # lnc: log tf, no idf, cosine-normalized (slide 7-Scoring s42)
            self.norm[cid] = math.sqrt(sum(log_tf(c) ** 2 for c in tf.values())) or 1.0
            for term, count in tf.items():
                self.postings[term][cid] = count
                added += 1

        self._champions = None
        return added

    def remove(self, chunk_ids: list[str]) -> int:
        """Purge these chunks entirely. Returns postings removed."""
        removed = 0
        for cid in chunk_ids:
            for term in self.forward.pop(cid, {}):
                self.postings[term].pop(cid, None)
                removed += 1
                if not self.postings[term]:  # term no longer occurs anywhere
                    del self.postings[term]
            self.doc_len.pop(cid, None)
            self.norm.pop(cid, None)
            self.chunk_doc.pop(cid, None)

        self._champions = None
        return removed

    # ------------------------------------------------------- champion lists
    @property
    def champions(self) -> dict[str, list[str]]:
        """Per term, the r chunks of highest tf — slide 8-Scoring s26.

        Rebuilt lazily after any mutation: r is fixed at build time, so the
        lists go stale the moment the postings change.
        """
        if self._champions is None:
            r = settings.champion_r
            self._champions = {
                term: sorted(posting, key=posting.get, reverse=True)[:r]
                for term, posting in self.postings.items()
            }
        return self._champions

    def candidates(self, terms, mode: str) -> set[str]:
        """Set A of contenders — slide 8-Scoring s18. Shared by VSM and BM25."""
        source = self.champions if mode == "champion" else self.postings
        return {cid for t in terms for cid in source.get(t, ())}

    # ------------------------------------------------------------ persistence
    def save(self):
        with open(INDEX_PATH, "wb") as fh:
            pickle.dump(
                {
                    "postings": dict(self.postings),
                    "forward": self.forward,
                    "doc_len": self.doc_len,
                    "norm": self.norm,
                    "chunk_doc": self.chunk_doc,
                },
                fh,
            )

    @classmethod
    def load(cls) -> "InvertedIndex":
        idx = cls()
        if INDEX_PATH.exists():
            with open(INDEX_PATH, "rb") as fh:
                state = pickle.load(fh)
            idx.postings = defaultdict(dict, state["postings"])
            idx.forward = state["forward"]
            idx.doc_len = state["doc_len"]
            idx.norm = state["norm"]
            idx.chunk_doc = state["chunk_doc"]
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
