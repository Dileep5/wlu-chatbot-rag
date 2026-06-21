import requests
import csv
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/index_old.php?cal=3&y=94"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

faculties = []

faculty_keywords = [
    "Faculty",
    "School",
    "College",
    "Lazaridis"
]

seen = set()

for link in soup.find_all("a"):

    href = link.get("href")
    text = link.get_text(strip=True)

    if not href:
        continue

    if "section.php" not in href:
        continue

    if not any(word in text for word in faculty_keywords):
        continue

    if text in seen:
        continue

    seen.add(text)

    full_url = "https://academic-calendar.wlu.ca/" + href

    faculties.append([text, full_url])

    print(text)

with open("outputs/faculties.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["faculty_name", "url"])

    writer.writerows(faculties)

print(f"\nSaved {len(faculties)} faculty links")