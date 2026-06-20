import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URLS_FILE = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/urls.txt")
OUTPUT_FILE = Path("/Users/deepu/Documents/Graduate Project/WLU ChatBot/outputs/raw_pages.csv")

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


def fetch_page(url: str) -> tuple[str, str]:
    """Download a webpage and return its title and cleaned text."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

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
    return title, cleaned_text


def main():
    urls = load_urls(URLS_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for url in urls:
        print(f"Scraping: {url}")
        try:
            title, text = fetch_page(url)
            rows.append({
                "url": url,
                "title": title,
                "text": text,
                "status": "success"
            })
            print(f"  -> Done: {title[:60]}")
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