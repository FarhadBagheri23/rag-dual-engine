"""The three retrieval engines, in one place.

Imported by api/search.py and by eval/, so "which engines exist" is stated
once. ponytail: a dict of functions, not a registry class with a decorator —
there are three of them and they are known at import time.
"""

from app.engines.lexical import vsm

ENGINES = {
    "vsm": vsm.search,
    # "bm25": bm25.search,   phase 3
    # "rag":  rag.search,    phase 4
}
