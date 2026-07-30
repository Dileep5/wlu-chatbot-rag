import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import quality_filter

BASE_DIR = Path(__file__).resolve().parent.parent
URLS_FILE = BASE_DIR / "urls.txt"
OUTPUT_FILE = BASE_DIR / "outputs" / "raw_pages.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def load_urls(file_path: Path) -> list[str]:
    """Read URLs from urls.txt."""
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    with file_path.open("r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    return urls


def fetch_page(url: str) -> tuple[str, str, str]:
    """Download a webpage and return its title, cleaned text, and a
    status - "success", or "filtered: <reason>" when the page fails one
    of quality_filter's post-fetch checks (wrong content-type, below the
    minimum content threshold). Raises on a real HTTP/network failure,
    exactly as before - only the return shape changed, to carry the
    filter outcome alongside the extracted content."""

    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if not quality_filter.is_acceptable_content_type(content_type):
        return "", "", f"filtered: non-HTML content-type ({content_type or 'unknown'})"

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy page elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else "No title"

    body = soup.body if soup.body else soup
    text = body.get_text(separator="\n", strip=True)

    # Clean up line breaks and extra spaces
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    # Remove duplicate consecutive lines
    cleaned_lines = []
    seen = set()
    for line in lines:
        if line not in seen:
            cleaned_lines.append(line)
            seen.add(line)

    cleaned_text = "\n".join(cleaned_lines)

    if not quality_filter.has_sufficient_content(cleaned_text):
        return title, cleaned_text, "filtered: below minimum content threshold"

    return title, cleaned_text, "success"


def main():
    urls = load_urls(URLS_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_content_fingerprints = set()

    for url in urls:
        print(f"Scraping: {url}")
        try:
            title, text, status = fetch_page(url)

            if status == "success":

                fingerprint = quality_filter.content_fingerprint(text)

                if fingerprint in seen_content_fingerprints:
                    status = "filtered: duplicate content"
                else:
                    seen_content_fingerprints.add(fingerprint)

            if status == "success":
                rows.append({
                    "url": url,
                    "title": title,
                    "text": text,
                    "status": status
                })
                print(f"  -> Done: {title[:60]}")
            else:
                rows.append({
                    "url": url,
                    "title": "",
                    "text": "",
                    "status": status
                })
                print(f"  -> Skipped: {status}")

        except Exception as e:
            rows.append({
                "url": url,
                "title": "",
                "text": "",
                "status": f"error: {e}"
            })
            print(f"  -> Failed: {e}")

        time.sleep(1)  # be polite to the website

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "text", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()