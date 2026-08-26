"""The three retrieval engines, in one place.

Imported by api/search.py and by eval/, so "which engines exist" is stated
once. ponytail: a dict of functions, not a registry class with a decorator —
there are three of them and they are known at import time.

Every engine has the signature (query, k, mode, prf) -> {"hits", "scored"}.
Only VSM honours `prf`: Rocchio is defined on the vector space model, and the
route rejects the combination rather than letting it silently no-op.
"""

from app.engines.lexical import bm25, vsm

ENGINES = {
    "vsm": vsm.search,
    "bm25": bm25.search,
    # "rag": rag.search,   phase 4
}

SUPPORTS_PRF = {"vsm"}
