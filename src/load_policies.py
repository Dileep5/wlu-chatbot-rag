"""Loads the lightweight Policies index (policy_number, policy_title,
source_url only - the policy body text is not duplicated here, it stays
vector-searchable via the normal crawl -> scrape -> clean -> chunk ->
ChromaDB pipeline like any other page).

Reads outputs/clean_pages.csv (produced by clean.py) rather than
re-scraping - every policy page crawler.py discovers under
/about/governance/assets/resources/ already goes through the same
scrape/clean pipeline as every other page; this just additionally
indexes the ones whose URL slug carries a real policy number
("12.2-student-code-of-conduct.html" -> "12.2"). Pages under that path
with no leading number (e.g. a "procedures-for-..." page) aren't
skipped from the corpus - they just don't get a structured index row,
since there's no policy number to store for them.
"""

import csv
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PAGES_FILE = BASE_DIR / "outputs" / "clean_pages.csv"
DB_PATH = BASE_DIR / "data" / "policies.db"

POLICY_PATH_PREFIX = "/about/governance/assets/resources/"
POLICY_NUMBER_PATTERN = re.compile(r"/(\d+\.\d+)-[^/]+\.html$")


def extract_policy_number(url):

    match = POLICY_NUMBER_PATTERN.search(url)

    return match.group(1) if match else None


def main():

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    inserted = 0

    with CLEAN_PAGES_FILE.open("r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row.get("status") != "success":
                continue

            url = row.get("url", "")

            if POLICY_PATH_PREFIX not in url:
                continue

            policy_number = extract_policy_number(url)

            if not policy_number:
                continue

            title = (row.get("title") or "").strip()

            if not title:
                continue

            cursor.execute(
                "INSERT OR IGNORE INTO policies "
                "(policy_number, policy_title, source_url) VALUES (?, ?, ?)",
                (policy_number, title, url)
            )

            inserted += cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Loaded {inserted} policies into {DB_PATH}")


if __name__ == "__main__":
    main()
