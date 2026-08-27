"""Tokenization and snippet extraction — slides 3-Document Preprocessing."""

import re

# [^\W_] is "word character except underscore", and Python's re is Unicode-aware
# by default — so this keeps Persian, Arabic, Cyrillic and CJK alongside ASCII.
# The older [a-z0-9]+ silently produced zero tokens for any non-Latin script,
# which made the lexical engines blind to those queries rather than merely bad
# at them.
TOKEN_RE = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, drop punctuation, split into alphanumeric tokens.

    Script-agnostic: .lower() is a no-op for scripts without case, so Persian
    and Arabic pass through unchanged.

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
