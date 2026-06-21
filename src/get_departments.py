import requests
import csv
from bs4 import BeautifulSoup

# Read faculties.csv
with open("faculties.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:

        faculty_name = row["faculty_name"]
        faculty_url = row["url"]

        print("\n" + "="*60)
        print("FACULTY:", faculty_name)
        print("="*60)

        response = requests.get(faculty_url)

        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a")

        for link in links:

            text = link.get_text(strip=True)
            href = link.get("href")

            if href:

                print(text, "->", href)