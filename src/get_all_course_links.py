import requests
import csv
import re
from bs4 import BeautifulSoup

course_rows = []

with open(
    "outputs/departments.csv",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        faculty_name = row["faculty_name"]
        department_name = row["department_name"]
        department_url = row["department_url"]

        print(f"\nScanning: {department_name}")

        try:

            response = requests.get(department_url)

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            found = 0

            for link in soup.find_all("a"):

                href = link.get("href")
                text = link.get_text(strip=True)

                if not href:
                    continue

                if "course.php" not in href:
                    continue

                full_url = (
                    "https://academic-calendar.wlu.ca/"
                    + href
                )

                course_code = text

                course_rows.append([
                    faculty_name,
                    department_name,
                    course_code,
                    full_url
                ])

                found += 1

            print(f"Found {found} courses")

        except Exception as e:

            print(
                f"Error in {department_name}"
            )

            print(e)

with open(
    "outputs/course_links.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "faculty_name",
        "department_name",
        "course_code",
        "course_url"
    ])

    writer.writerows(course_rows)

print(
    f"\nSaved {len(course_rows)} course links"
)