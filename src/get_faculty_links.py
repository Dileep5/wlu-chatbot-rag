import requests
import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Confirmed working faculty-and-staff.html hub pages (verified live). Not
# every faculty listed in the academic calendar has one - e.g. School of
# International Policy and Governance returns a 404 - so this scraper is
# tolerant of missing/failed pages rather than assuming full coverage.
FACULTY_HUB_URLS = [
    ("Faculty of Arts", "https://www.wlu.ca/academics/faculties/faculty-of-arts/faculty-and-staff.html"),
    ("Faculty of Education", "https://www.wlu.ca/academics/faculties/faculty-of-education/faculty-and-staff.html"),
    ("Faculty of Human and Social Sciences", "https://www.wlu.ca/academics/faculties/faculty-of-human-and-social-sciences/faculty-and-staff.html"),
    ("Faculty of Liberal Arts", "https://www.wlu.ca/academics/faculties/faculty-of-liberal-arts/faculty-and-staff.html"),
    ("Faculty of Music", "https://www.wlu.ca/academics/faculties/faculty-of-music/faculty-and-staff.html"),
    ("Faculty of Science", "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-and-staff.html"),
    ("Faculty of Social Work", "https://www.wlu.ca/academics/faculties/faculty-of-social-work/faculty-and-staff.html"),
    ("Lazaridis School of Business and Economics", "https://www.wlu.ca/academics/faculties/lazaridis-school-of-business-and-economics/faculty-and-staff.html"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def scrape_faculty_links(output_path):

    # Keyed by resolved profile URL so the same profile is emitted only
    # once, no matter how many times it's linked on the hub page(s).
    # Some hub pages have stray, empty-text <a> tags left over from CMS
    # editing that point at a real profile URL but carry no name - those
    # are skipped before ever being registered here, which is what
    # prevents them from creating duplicate/blank rows. Genuine cross-
    # listings (the same person under multiple departments) still merge
    # into a single row, keeping every distinct faculty/department value.
    profiles = {}
    order = []

    for faculty_name, hub_url in FACULTY_HUB_URLS:

        print(f"\nScanning: {faculty_name}")

        try:
            response = requests.get(hub_url, headers=HEADERS, timeout=20)

            if response.status_code != 200:
                print(f"  Skipped (status {response.status_code})")
                continue

        except Exception as e:
            print(f"  Error: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Hub pages group people under <h2> department headings and <h3>
        # category headings (Faculty / Adjunct Faculty / Staff / ...).
        # Only categories whose heading mentions "Faculty" are academic
        # faculty members - Staff/Postdoctoral Fellows etc. are excluded.
        current_department = None
        current_group = ""
        found = 0

        for el in soup.find_all(["h2", "h3", "a"]):

            if el.name == "h2":
                current_department = el.get_text(strip=True)

            elif el.name == "h3":
                current_group = el.get_text(strip=True)

            elif el.name == "a":

                href = el.get("href")

                if not href or "faculty-profiles" not in href:
                    continue

                if "faculty" not in current_group.lower():
                    continue

                if not current_department:
                    continue

                link_text = el.get_text(strip=True)

                if not link_text:
                    continue

                full_url = urljoin(hub_url, href)

                if full_url not in profiles:
                    profiles[full_url] = {
                        "member_name": link_text,
                        "faculty_names": [],
                        "department_names": [],
                    }
                    order.append(full_url)

                entry = profiles[full_url]

                if faculty_name not in entry["faculty_names"]:
                    entry["faculty_names"].append(faculty_name)

                if current_department not in entry["department_names"]:
                    entry["department_names"].append(current_department)

                found += 1

        print(f"  Found {found} faculty links")

    rows = [
        [
            " | ".join(profiles[url]["faculty_names"]),
            " | ".join(profiles[url]["department_names"]),
            profiles[url]["member_name"],
            url,
        ]
        for url in order
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "faculty_name",
            "department_name",
            "member_name",
            "profile_url"
        ])

        writer.writerows(rows)

    print(f"\nSaved {len(rows)} unique faculty profiles")

    return rows


if __name__ == "__main__":
    scrape_faculty_links(output_path="outputs/faculty_links.csv")
