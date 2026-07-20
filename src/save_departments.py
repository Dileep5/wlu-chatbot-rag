import requests
import csv
from bs4 import BeautifulSoup


def scrape_departments(faculties_csv, output_path):

    departments = []
    seen_urls = set()
    to_expand = []

    with open(faculties_csv, newline="", encoding="utf-8") as f:

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

                    if full_url in seen_urls:
                        continue

                    seen_urls.add(full_url)

                    departments.append([
                        faculty_name,
                        text,
                        full_url
                    ])

                    to_expand.append((faculty_name, text, full_url))

                    print("  ", text)

    # Some department pages are umbrella/overview pages for a combined
    # subject area (e.g. "Computer Science and Physics") and link out to
    # their real sub-departments one hop deeper, using the same
    # department.php pattern. Follow those links too, generically, so
    # sub-departments (and their courses) aren't missed.
    while to_expand:

        faculty_name, department_name, department_url = to_expand.pop(0)

        try:
            response = requests.get(department_url)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Error expanding {department_name}: {e}")
            continue

        for link in soup.find_all("a"):

            text = link.get_text(strip=True)
            href = link.get("href")

            if not href or "department.php" not in href:
                continue

            full_url = "https://academic-calendar.wlu.ca/" + href

            if full_url in seen_urls:
                continue

            seen_urls.add(full_url)

            departments.append([
                faculty_name,
                text,
                full_url
            ])

            to_expand.append((faculty_name, text, full_url))

            print(f"   (sub-department of {department_name}):", text)

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "faculty_name",
            "department_name",
            "department_url"
        ])

        writer.writerows(departments)

    print(f"\nSaved {len(departments)} departments")

    return departments


if __name__ == "__main__":
    scrape_departments(
        faculties_csv="outputs/faculties.csv",
        output_path="outputs/departments.csv"
    )
