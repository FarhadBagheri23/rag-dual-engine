"""Word -> term: the analysis pipeline — slides 3-Document Preprocessing.

Slide 5 draws the distinction this module implements. A *word* is a delimited
string as it appears in the text; a *term* is a normalized word, an equivalence
class. Everything here exists to collapse words into terms, in the order the
slides introduce them:

    s11  normalize to terms   delete periods (U.S.A. -> USA) and hyphens
                              (anti-discriminatory -> antidiscriminatory)
    s22  other languages      de-accent (résumé -> resume, Tübingen -> Tubingen)
    s14  case folding         lower()
    s6   tokenization         Unicode-aware, script-agnostic
    s17  stemming             Porter, the algorithm named on s18

Slide 11 is emphatic: "Crucial: Need to normalize indexed text as well as query
terms into the same form." That is why `tokenize` is the single path — index
writes, VSM, BM25, Rocchio and snippet highlighting all call it, so there is no
second place where the two sides could drift into different equivalence classes.
"""

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

# [^\W_] is "word character except underscore", and Python's re is Unicode-aware
# by default — so this keeps Persian, Arabic, Cyrillic and CJK alongside ASCII.
# The older [a-z0-9]+ silently produced zero tokens for any non-Latin script,
# which made the lexical engines blind to those queries rather than merely bad
# at them.
TOKEN_RE = re.compile(r"[^\W_]+")

# s11, "deleting periods to form a term": U.S.A., USA -> USA. Restricted to
# acronym runs — a letter-then-dot, twice or more — because the general rule
# "delete every period" also welds together the sentence boundaries PDF text
# extraction leaves unspaced ("...the index.The next section" -> "indexthe").
_ACRONYM_RE = re.compile(r"(?:(?<![^\W_])[^\W_]\.){2,}")

# s11, "deleting hyphens to form a term": anti-discriminatory -> antidiscriminatory.
# Only between two word characters, so a dash used as punctuation is left to the
# tokenizer to drop.
_HYPHEN_RE = re.compile(r"(?<=[^\W_])-(?=[^\W_])")

# s18 names Porter's algorithm specifically. Snowball ships it under that name;
# "english" would be Porter2, a later revision the slides do not cover.
_stemmer = snowballstemmer.stemmer("porter")


@lru_cache(maxsize=1 << 17)
def stem(token: str) -> str:
    """Porter stemming — s17, "crude affix chopping".

    s17 also notes the exact stemmed form does not matter, only the equivalence
    class it induces: organize/organizes/organizing all collapse to `organ`,
    which is not a word and does not need to be.

    Cached because token frequency is Zipfian — a corpus of hundreds of
    thousands of tokens resolves to a few thousand distinct stem calls, and the
    same handful of query terms is stemmed on every single search.
    """
    return _stemmer.stemWord(token)


def _deaccent(text: str) -> str:
    """s22: résumé -> resume, Tübingen -> Tubingen.

    The slide's criterion is "how are your users likely to write their queries"
    — users often omit accents even in languages that require them, so the
    de-accented form is the one both sides should meet at.

    NFKD splits a letter from its combining marks, so dropping the marks removes
    accents, umlauts and Arabic harakat while leaving base letters and every
    unaccented script untouched. The compatibility half also unpicks the
    ligatures PDF extraction emits, so `ﬁle` indexes as `file`.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize(text: str) -> str:
    """Words -> normalized characters, before tokenization splits them.

    Order matters: the periods and hyphens must go while they still sit between
    the characters they join, since the tokenizer would otherwise have already
    split the term at exactly those points.
    """
    text = _ACRONYM_RE.sub(lambda m: m.group().replace(".", ""), text)
    text = _HYPHEN_RE.sub("", text)
    return _deaccent(text).lower()


def tokenize(text: str) -> list[str]:
    """Text -> the terms an index or a query is built from.

    No stopword list is applied. Slide 10 notes the trend is away from removing
    stopwords, and this corpus is mutable — a list derived from collection
    frequency would go stale on every upload. Stopword-like terms are instead
    dropped at query time by index elimination on idf (slide 8-Scoring s24),
    which needs no stored list and adapts as the corpus changes.

    Stemming is applied, on the other hand, because s20 is only ambivalent about
    it for English ("helps recall but harms precision") and unambiguous for the
    morphologically richer languages this corpus can also hold.
    """
    return [stem(t) for t in TOKEN_RE.findall(normalize(text))]


def snippet(text: str, query_terms: set[str], window: int = 45) -> str:
    """A window of `text` centred on its densest run of query terms."""
    words = text.split()
    if not words:
        return ""

    # tokenize(), not a bare regex: query_terms are stemmed, so the surface word
    # has to travel through the same pipeline before the two can be compared.
    hits = [i for i, w in enumerate(words) if query_terms & set(tokenize(w))]
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
