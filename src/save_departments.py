import requests
import csv
from bs4 import BeautifulSoup

departments = []

with open("outputs/faculties.csv", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        faculty_name = row["faculty_name"]
        faculty_url = row["url"]

        print("Scanning:", faculty_name)

        response = requests.get(faculty_url)

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a"):

            text = link.get_text(strip=True)
            href = link.get("href")

            if href and "department.php" in href:

                full_url = "https://academic-calendar.wlu.ca/" + href

                departments.append([
                    faculty_name,
                    text,
                    full_url
                ])

                print("  ", text)

with open("outputs/departments.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "faculty_name",
        "department_name",
        "department_url"
    ])

    writer.writerows(departments)

print(f"\nSaved {len(departments)} departments")