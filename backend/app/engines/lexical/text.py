"""Tokenization and snippet extraction — slides 3-Document Preprocessing."""

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, drop punctuation, split into alphanumeric tokens.

    No stopword list is applied at index time. Slide 10 notes the trend is away
    from removing stopwords, and this corpus is mutable — a list derived from
    collection frequency would go stale on every upload. Stopword-like terms
    are instead dropped at query time by index elimination on idf (slide 24),
    which needs no stored list and adapts as the corpus changes.
    """
    return TOKEN_RE.findall(text.lower())


def snippet(text: str, query_terms: set[str], window: int = 45) -> str:
    """A window of `text` centred on its densest run of query terms."""
    words = text.split()
    if not words:
        return ""

    hits = [i for i, w in enumerate(words) if TOKEN_RE.findall(w.lower()) and
            set(TOKEN_RE.findall(w.lower())) & query_terms]
    if not hits:
        return " ".join(words[:window]) + ("…" if len(words) > window else "")

    # slide the window to wherever the most query terms fall
    best_start, best_hits = 0, -1
    for start in range(0, max(1, len(words) - window + 1)):
        n = sum(1 for h in hits if start <= h < start + window)
        if n > best_hits:
            best_start, best_hits = start, n

    piece = words[best_start:best_start + window]
    return ("… " if best_start else "") + " ".join(piece) + (" …" if best_start + window < len(words) else "")
