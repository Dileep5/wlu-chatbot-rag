"""Modular quality-filter stage for the ingestion pipeline (Phase 2).

Owns every filtering policy - robots.txt compliance, URL normalization,
duplicate removal, minimum content threshold, and content-type filtering
- in one place, rather than each embedded inline inside crawler.py and
scrape.py. Those two scripts call into this module at the point where
the relevant data actually becomes available:

  crawler.py  -> filter_urls()               (pre-fetch, URL-level)
  scrape.py   -> is_acceptable_content_type() (needs the real response)
              -> has_sufficient_content()     (needs the extracted text)
              -> content_fingerprint()        (needs the extracted text)
"""

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests

# -----------------------------
# URL normalization - the canonical form used as the dedup key
# everywhere in this module: lowercase scheme+host, fragment stripped,
# query string stripped (every page in scope for this pipeline is
# static informational HTML, not a search/filter view), and a trailing
# "index.html" segment treated the same as the directory it lives in.
# -----------------------------

def normalize_url(url):

    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"

    if path.endswith("/index.html"):
        path = path[: -len("index.html")]

    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path = path + "/"

    return urlunsplit((scheme, netloc, path, "", ""))


# -----------------------------
# robots.txt compliance - one RobotFileParser per domain, cached so a
# large URL list doesn't refetch robots.txt per URL.
# -----------------------------

_robots_cache = {}


def _get_robot_parser(netloc):

    if netloc not in _robots_cache:

        parser = RobotFileParser()

        # Fetched via requests, not RobotFileParser.read()'s own urllib
        # fetch: read() was found to silently fail in this environment
        # with a local SSL trust-store error (self-signed cert in the
        # chain via urllib's SSL context) even though the exact same
        # https:// URL fetches fine through requests (which every other
        # network call in this pipeline already uses successfully). That
        # failure was being caught below and treated as "fail open" -
        # correct for a genuinely missing robots.txt (confirmed for
        # events.wlu.ca, a real 404), but wrong here, since it silently
        # disabled robots.txt compliance entirely rather than only for
        # the sites that truly have none.
        try:
            response = requests.get(f"https://{netloc}/robots.txt", timeout=10)

            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None

        except Exception:
            parser = None

        _robots_cache[netloc] = parser

    return _robots_cache[netloc]


def is_allowed_by_robots(url, user_agent="*"):

    parser = _get_robot_parser(urlsplit(url).netloc)

    if parser is None:
        return True

    return parser.can_fetch(user_agent, url)


# -----------------------------
# URL-level filtering - normalizes, dedups, and applies robots.txt to a
# raw URL list. Used by crawler.py as the final step before writing
# urls.txt, regardless of which discovery pass (BFS, sitemap-driven,
# or events) a given URL came from.
# -----------------------------

def filter_urls(urls):

    seen_normalized = set()
    kept = []

    for url in urls:

        normalized = normalize_url(url)

        if normalized in seen_normalized:
            continue

        if not is_allowed_by_robots(url):
            continue

        seen_normalized.add(normalized)
        kept.append(url)

    return kept


# -----------------------------
# Post-fetch content-quality filtering - used by scrape.py once a page
# has actually been fetched, since these checks need the real response
# (Content-Type header) or the extracted body text.
# -----------------------------

_MIN_CONTENT_WORDS = 150


def is_acceptable_content_type(content_type):

    if not content_type:
        return False

    return content_type.split(";")[0].strip().lower() == "text/html"


def has_sufficient_content(text):

    if not text:
        return False

    return len(text.split()) >= _MIN_CONTENT_WORDS


# -----------------------------
# Content-level duplicate detection - a normalized-text fingerprint so
# two different URLs whose actual page content is substantially the
# same (print-friendly variants, redirected duplicates the crawler
# discovered under two different paths) don't both get scraped, chunked,
# and embedded. The caller keeps a running set of fingerprints already
# seen this run and skips any page whose fingerprint repeats.
# -----------------------------

def content_fingerprint(text):

    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
