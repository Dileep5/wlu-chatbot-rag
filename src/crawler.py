import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import quality_filter

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_METADATA_FILE = BASE_DIR / "data" / "corpus_metadata.json"


def _write_corpus_retrieval_timestamp():
    """Phase 3: stamps data/corpus_metadata.json with the moment this
    ingestion run started fetching from the live WLU site - the single
    source of truth citation.py reads "Retrieval Date" from for every
    response. Written here (ingestion), not in retriever.py, so this
    never touches structured/hybrid retrieval logic; generated fresh
    every run, never hardcoded."""

    CORPUS_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CORPUS_METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {"retrieved_at": datetime.now(timezone.utc).isoformat()},
            f
        )


# -----------------------------
# Pass 1: academics BFS crawl. Unchanged from before Phase 2 - same seed,
# same 100-page cap, same link-validity rule. This is the corpus the
# existing 226-check regression suite and every deterministic
# structured-retrieval path were originally built and validated against.
# -----------------------------

BFS_START_URL = "https://www.wlu.ca/academics/index.html"
BFS_MAX_PAGES = 100


def _is_valid_bfs_link(url):

    parsed = urlparse(url)

    return (
        parsed.netloc == "www.wlu.ca"
        and url.startswith("https://www.wlu.ca")
    )


def crawl_academics_bfs():

    visited = set()
    to_visit = [BFS_START_URL]

    while to_visit and len(visited) < BFS_MAX_PAGES:

        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        print(f"Crawling: {current_url}")

        try:

            response = requests.get(current_url, timeout=10)

            if response.status_code != 200:
                continue

            visited.add(current_url)

            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):

                full_url = urljoin(current_url, link["href"])

                if (
                    _is_valid_bfs_link(full_url)
                    and full_url not in visited
                    and full_url not in to_visit
                ):
                    to_visit.append(full_url)

        except Exception as e:

            print("Error:", e)

    return visited


# -----------------------------
# Pass 2 (Phase 2): sitemap-driven section discovery. Both www.wlu.ca and
# students.wlu.ca publish real sitemap.xml files (confirmed via each
# domain's robots.txt Sitemap: directive) - filtering their <loc> entries
# by an explicit path-prefix allowlist per section is precise, curated
# discovery, not link-following. This is what lets the crawler reach
# policies, deadlines, campus services, and student services at all: all
# of them live on students.wlu.ca, a domain the original BFS crawl never
# visits (confirmed by curl: the www.wlu.ca deadlines URL 302-redirects
# away; the real page is 200 OK only on students.wlu.ca).
# -----------------------------

SITEMAP_NAMESPACE = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

SITEMAP_DOMAINS = ("www.wlu.ca", "students.wlu.ca")

# (domain, path prefix) - one row per target section. "/academics/
# support-and-advising/" deliberately covers its own
# "calendars-and-petitions/" subpath (academic deadlines/calendars) too,
# so no separate entry is needed for that.
#
# Coverage audit (2026-08) found two entire sections missing from this
# list, not just a few individual pages within an already-covered one:
# "/academics/convocation/" (graduation dates/deadlines, applying to
# graduate, ceremony/gown-rental logistics) and "/finances/"
# (scholarships-and-bursaries, tuition-and-fees fee breakdowns with
# actual dollar figures, financial-aid, graduate-funding-and-awards) -
# confirmed live, every page under both is real and publicly served
# (200 OK) but neither prefix was ever in this allowlist, so Pass 2
# never had a chance to discover any of it, regardless of how many
# individual URLs happened to get added elsewhere by hand. Before this
# fix, the only scholarship-related content in the corpus was "Writing
# the Scholarship Proposal" (support-and-advising) and the only
# tuition content was policy text about the tuition guarantee, never
# an actual fee schedule - both confirmed by a real user-facing query
# citing the wrong page as if it were the answer.
SITEMAP_SECTIONS = [
    ("www.wlu.ca", "/about/governance/assets/resources/"),   # policies
    ("students.wlu.ca", "/academics/support-and-advising/"), # deadlines/calendars + advising
    ("students.wlu.ca", "/academics/convocation/"),           # convocation/graduation
    ("students.wlu.ca", "/campus-services/"),                # campus services
    ("students.wlu.ca", "/support-and-wellness/"),            # student services/wellness
    ("students.wlu.ca", "/finances/"),                        # scholarships/tuition/financial aid
]

# Within the policies path specifically, these aren't part of the
# numbered policy library itself (annual reports and a news self-study
# piece that merely mention "policy" in their slug) - excluded so the
# corpus stays the actual policy documents, not tangential coverage.
_POLICY_EXCLUDE_KEYWORDS = ("annual-report", "self-study")

FAQ_KEYWORD = "faq"

NEWS_PATH_PREFIX = "/news/"
NEWS_KEEP_COUNT = 75  # within the requested 50-100 range


def _fetch_sitemap_entries(domain):
    """[{'url': ..., 'lastmod': ...}, ...] for one domain's sitemap.xml,
    or [] if it can't be fetched/parsed - a missing sitemap degrades to
    "this domain contributes nothing to Pass 2", not a crawl failure."""

    try:
        response = requests.get(f"https://{domain}/sitemap.xml", timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as e:
        print(f"Could not fetch/parse sitemap for {domain}: {e}")
        return []

    entries = []

    for url_element in root.findall(f"{SITEMAP_NAMESPACE}url"):

        loc = url_element.find(f"{SITEMAP_NAMESPACE}loc")

        if loc is None or not (loc.text or "").strip():
            continue

        lastmod = url_element.find(f"{SITEMAP_NAMESPACE}lastmod")

        entries.append({
            "url": (loc.text or "").strip(),
            "lastmod": (lastmod.text or "").strip() if lastmod is not None else "",
        })

    return entries


def discover_section_and_faq_urls(entries_by_domain):

    discovered = set()

    for domain, prefix in SITEMAP_SECTIONS:

        for entry in entries_by_domain.get(domain, []):

            path = urlparse(entry["url"]).path

            if not path.startswith(prefix):
                continue

            if prefix == "/about/governance/assets/resources/" and any(
                keyword in path for keyword in _POLICY_EXCLUDE_KEYWORDS
            ):
                continue

            discovered.add(entry["url"])

    for domain in SITEMAP_DOMAINS:

        for entry in entries_by_domain.get(domain, []):

            if FAQ_KEYWORD in entry["url"].lower():
                discovered.add(entry["url"])

    return discovered


def discover_recent_news_urls(entries_by_domain):
    """Latest NEWS_KEEP_COUNT news articles by sitemap <lastmod>, not the
    full news archive - historical news dilutes vector-search relevance
    (confirmed directly: a "What scholarships are available?" query was
    dragged toward an individual student's years-old personal scholarship
    story instead of general scholarship info, purely because it was the
    only content that superficially matched)."""

    news_entries = [
        entry
        for entry in entries_by_domain.get("www.wlu.ca", [])
        if urlparse(entry["url"]).path.startswith(NEWS_PATH_PREFIX) and entry["lastmod"]
    ]

    news_entries.sort(key=lambda entry: entry["lastmod"], reverse=True)

    return {entry["url"] for entry in news_entries[:NEWS_KEEP_COUNT]}


# -----------------------------
# Pass 3: events.wlu.ca. No sitemap.xml exists there (confirmed: 404), so
# this is a small, tightly-capped BFS scoped entirely to that one
# subdomain. A live events calendar's homepage/listing pages naturally
# surface currently-scheduled (upcoming) events rather than a historical
# archive, so this satisfies "keep upcoming events" without needing
# separate date parsing.
# -----------------------------

EVENTS_START_URL = "https://events.wlu.ca/index.html"
EVENTS_MAX_PAGES = 40


def _is_valid_events_link(url):

    parsed = urlparse(url)

    return parsed.netloc == "events.wlu.ca" and url.startswith("https://events.wlu.ca")


def crawl_upcoming_events():

    visited = set()
    to_visit = [EVENTS_START_URL]

    while to_visit and len(visited) < EVENTS_MAX_PAGES:

        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        try:

            response = requests.get(current_url, timeout=10)

            if response.status_code != 200:
                continue

            visited.add(current_url)

            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):

                full_url = urljoin(current_url, link["href"])

                if (
                    _is_valid_events_link(full_url)
                    and full_url not in visited
                    and full_url not in to_visit
                ):
                    to_visit.append(full_url)

        except Exception as e:

            print("Error:", e)

    return visited


def main():

    _write_corpus_retrieval_timestamp()

    print("Pass 1: academics BFS crawl...")
    bfs_urls = crawl_academics_bfs()
    print(f"  -> {len(bfs_urls)} pages")

    print("Pass 2: sitemap-driven section discovery (policies, deadlines, "
          "campus services, student services, FAQs, recent news)...")
    entries_by_domain = {
        domain: _fetch_sitemap_entries(domain) for domain in SITEMAP_DOMAINS
    }
    section_urls = discover_section_and_faq_urls(entries_by_domain)
    news_urls = discover_recent_news_urls(entries_by_domain)
    print(f"  -> {len(section_urls)} section/FAQ pages, {len(news_urls)} recent news pages")

    print("Pass 3: events.wlu.ca crawl (no sitemap available there)...")
    event_urls = crawl_upcoming_events()
    print(f"  -> {len(event_urls)} pages")

    all_urls = bfs_urls | section_urls | news_urls | event_urls

    print(f"\n{len(all_urls)} URLs discovered before quality filtering.")

    filtered_urls = quality_filter.filter_urls(sorted(all_urls))

    print(f"{len(filtered_urls)} URLs remain after robots.txt + "
          f"normalization + dedup filtering.")

    with open("urls.txt", "w") as f:
        for url in filtered_urls:
            f.write(url + "\n")

    print(f"\nSaved {len(filtered_urls)} URLs to urls.txt")


if __name__ == "__main__":
    main()
