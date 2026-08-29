"""Web results as extra RAG passages — the corpus, temporarily widened.

Nothing here is a second retrieval *model*. A web result is turned into the
same shape a Chroma chunk has — a title and a span of text — and handed to
rag.py, which numbers it alongside the local passages and lets the existing
[Doc N] machinery cite it. That is the whole integration: one numbered context,
one citation format, one prompt.

    ponytail: DuckDuckGo's HTML endpoint, parsed with `re`. No API key, no new
    dependency — httpx already ships with the openai client, and the rest is
    stdlib. The ceiling is real and worth naming: this is scraping, so a markup
    change at DuckDuckGo breaks it, and there is no rate-limit contract. If it
    ever needs to be reliable, `search()` is the only function to replace —
    swap in Brave/Tavily/Serper behind the same return shape and nothing above
    this line changes.

Snippets only, not page bodies. Fetching and boilerplate-stripping every result
would mean another dependency, several seconds of latency, and a new class of
failure, to enrich passages the model mostly uses for orientation anyway.
    ponytail: fetch full pages when the snippets are measurably too thin.
"""

import html
import re
import urllib.parse

import httpx

from app.core.config import settings

ENDPOINT = "https://html.duckduckgo.com/html/"

# The endpoint serves a bare "please enable JS" page to clients it does not
# recognise, so it needs a browser UA to answer at all.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

# A result is a title anchor followed by a snippet anchor. Pairing them in one
# pattern rather than collecting two lists keeps a result whose snippet is
# missing from silently stealing the next result's text.
RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)

TAG_RE = re.compile(r"<[^>]+>")  # snippets arrive with <b> around the matches


def _clean(fragment: str) -> str:
    """Markup fragment -> plain text."""
    return html.unescape(TAG_RE.sub("", fragment)).strip()


def _direct(url: str) -> str:
    """Unwrap DuckDuckGo's redirect, when it uses one.

    Some results come back as //duckduckgo.com/l/?uddg=<encoded>. A citation
    has to point at the page, not at a tracker, or the reader cannot tell one
    source from another.
    """
    if "duckduckgo.com/l/" not in url:
        return url
    target = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("uddg")
    return urllib.parse.unquote(target[0]) if target else url


def domain(url: str) -> str:
    """`en.wikipedia.org` from a full URL — what the UI shows under a source."""
    return urllib.parse.urlparse(url).netloc.removeprefix("www.")


def search(query: str, k: int | None = None) -> list[dict]:
    """Top web results as passages: [{title, url, text}].

    Returns [] on any failure. Web search is an augmentation, never a hard
    dependency — if it is down, the answer should still come from the corpus
    rather than the whole request failing.
    """
    k = k or settings.web_top_k
    try:
        response = httpx.post(
            ENDPOINT,
            data={"q": query},
            headers=HEADERS,
            timeout=settings.web_timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception:
        return []

    out, seen = [], set()
    for match in RESULT_RE.finditer(response.text):
        url = _direct(html.unescape(match["url"]))
        text = _clean(match["snippet"])
        # One passage per site. Three pages of the same domain crowd out the
        # disagreement that makes several sources worth having.
        host = domain(url)
        if not text or host in seen:
            continue
        seen.add(host)
        out.append({"title": _clean(match["title"]) or host, "url": url, "text": text})
        if len(out) >= k:
            break
    return out
