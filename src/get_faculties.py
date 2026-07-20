import requests
import csv
from bs4 import BeautifulSoup

FACULTY_KEYWORDS = [
    "Faculty",
    "School",
    "College",
    "Lazaridis"
]


def scrape_faculties(cal, year, output_path):

    url = f"https://academic-calendar.wlu.ca/index_old.php?cal={cal}&y={year}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    faculties = []
    seen = set()

    for link in soup.find_all("a"):

        href = link.get("href")
        text = link.get_text(strip=True)

        if not href:
            continue

        if "section.php" not in href:
            continue

        if not any(word in text for word in FACULTY_KEYWORDS):
            continue

        if text in seen:
            continue

        seen.add(text)

        full_url = "https://academic-calendar.wlu.ca/" + href

        faculties.append([text, full_url])

        print(text)

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["faculty_name", "url"])

        writer.writerows(faculties)

    print(f"\nSaved {len(faculties)} faculty links")

    return faculties


if __name__ == "__main__":
    scrape_faculties(cal=3, year=94, output_path="outputs/faculties.csv")
