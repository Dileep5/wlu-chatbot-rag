import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

START_URL = "https://www.wlu.ca/academics/index.html"

MAX_PAGES = 100

visited = set()
to_visit = [START_URL]


def is_valid_wlu_link(url):

    parsed = urlparse(url)

    return (
        parsed.netloc == "www.wlu.ca"
        and url.startswith("https://www.wlu.ca")
    )


while to_visit and len(visited) < MAX_PAGES:

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
                is_valid_wlu_link(full_url)
                and full_url not in visited
                and full_url not in to_visit
            ):
                to_visit.append(full_url)

    except Exception as e:

        print("Error:", e)

with open("urls.txt", "w") as f:

    for url in sorted(visited):
        f.write(url + "\n")

print(f"\nSaved {len(visited)} URLs to urls.txt")