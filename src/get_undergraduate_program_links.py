import csv
import re
import requests
from bs4 import BeautifulSoup

# Discovery strategy confirmed live during Sprint 11A: undergraduate
# programs are NOT reachable through the search.php?...&s_pdt=N degree-
# type search (that returns department pages, not program pages -
# confirmed live, which is why the old undergraduate_programs.csv only
# ever produced generic degree-type shells like "General BSc without
# Designation"). The reliable path is department page -> the page's own
# <a name="Programs"> anchor -> its next-sibling div.reqs -> program.php
# links inside it - the exact same anchor+div.reqs shape already proven
# for graduate "Program_Requirements" extraction (Sprint 7D), just a
# different anchor name. Verified against a random sample and against
# all 120 undergraduate departments during the Sprint 11A investigation.

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://academic-calendar.wlu.ca/"


def _extract_program_links(department_soup):

    anchor = department_soup.find("a", attrs={"name": "Programs"})

    if not anchor:
        return []

    reqs_div = anchor.find_next_sibling("div", class_="reqs")

    if not reqs_div:
        return []

    links = []

    for link in reqs_div.find_all("a", href=True):

        if "program.php" not in link["href"]:
            continue

        name = link.get_text(strip=True)

        if not name:
            continue

        href = link["href"]

        url = href if href.startswith("http") else BASE_URL + href

        links.append((name, url))

    return links


def get_undergraduate_program_links(
    departments_csv="outputs/undergraduate_departments.csv",
    output_csv="outputs/undergraduate_program_links.csv",
):

    # Keyed by URL (the stable identifier a program.php link resolves
    # to) so a joint/combined program discovered on more than one
    # department's page - e.g. "Honours BSc in Computer Science and
    # Honours Bachelor of Business Administration" appears on both the
    # Computer Science and Business department pages - is written to
    # the output exactly once, not once per department that links to it.
    programs_by_url = {}

    with open(departments_csv, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            department_name = row["department_name"]
            department_url = row["department_url"]

            print(f"\nScanning: {department_name}")

            try:

                response = requests.get(
                    department_url, headers=HEADERS, timeout=20
                )

                if response.status_code != 200:
                    print(f"  Failed ({response.status_code})")
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                links = _extract_program_links(soup)

                print(f"  Found {len(links)} program link(s)")

                for name, url in links:

                    if url not in programs_by_url:
                        programs_by_url[url] = name

            except Exception as e:
                print(f"  Error: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        writer.writerow(["program_name", "program_url"])

        for url, name in sorted(programs_by_url.items(), key=lambda item: item[1]):
            writer.writerow([name, url])

    print(f"\nDone! {len(programs_by_url)} unique undergraduate program(s) found.")

    return programs_by_url


if __name__ == "__main__":
    get_undergraduate_program_links()
