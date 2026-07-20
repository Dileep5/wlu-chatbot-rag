import requests
import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Behavior-driven configuration for each hub page's heading structure.
# Rather than the walking loop hardcoding "h2 = department, h3 =
# category" as a fixed assumption, each hub page declares where these
# values actually come from - so the loop itself never branches on
# which faculty it's parsing, only on these parameters.
#
#   category_source:    "h2" | "h3" | "none"
#       Which heading level carries the "is this a real faculty member"
#       signal. "none" accepts every entry with no category check at all.
#   category_keywords:   list[str]
#       Case-insensitive substrings, any one of which makes the category
#       text count as a real faculty member (e.g. "Adjunct Faculty" and
#       "Dean and Associate Deans" can both pass, depending on what's in
#       this list). Most pages only need ["faculty"].
#   department_source:   "h2" | "h3" | "faculty_name"
#       Which heading level supplies the raw department text, or
#       "faculty_name" if the page offers no department breakdown at all.
#   department_parse:    "as_is" | "split_on_comma"
#       How the raw department text is turned into a department name.
#       "split_on_comma" takes the text after the first comma, for pages
#       that fuse category and department into a single heading.
#   department_fallback: "faculty_name" | None
#       What to use if the sourced/parsed department comes out empty.
#       None means reject the entry entirely.
_STANDARD_CONFIG = {
    "category_source": "h3",
    "category_keywords": ["faculty"],
    "department_source": "h2",
    "department_parse": "as_is",
    "department_fallback": None,
}

# Same shape as _STANDARD_CONFIG, but this page also uses a "Deans"
# category (distinct from "Faculty") for its Dean's Office listing -
# verified live that everyone under it holds a genuine Professor/Dean
# title, same as the "Faculty" category everywhere else.
_STANDARD_CONFIG_WITH_LEADERSHIP_CATEGORIES = {
    "category_source": "h3",
    "category_keywords": ["faculty", "dean", "program director"],
    "department_source": "h2",
    "department_parse": "as_is",
    "department_fallback": None,
}

# Some hub pages have no per-department breakdown at all (no <h2> is ever
# present) - every entry is attributed directly to the overall faculty
# instead. The category signal is unchanged from the standard case; it
# still comes from <h3> ("Full-time Faculty" / "Contract Teaching
# Faculty" pass, "Staff" is correctly excluded).
_NO_DEPARTMENT_BREAKDOWN_CONFIG = {
    "category_source": "h3",
    "category_keywords": ["faculty"],
    "department_source": "faculty_name",
    "department_parse": "as_is",
    "department_fallback": "faculty_name",
}

# The mirror image of the standard case: category signal at h2 ("Faculty"
# vs "Staff" vs "Leadership" vs "Dean and Associate Deans"), with the
# real department/specialization breakdown at h3 ("Composition",
# "Community Music", "Music Therapy", ...) instead of h2.
_CATEGORY_H2_DEPARTMENT_H3_CONFIG = {
    "category_source": "h2",
    "category_keywords": ["faculty"],
    "department_source": "h3",
    "department_parse": "as_is",
    "department_fallback": "faculty_name",
}

# Category and department are fused into a single h2 heading following a
# "Category, Department" comma convention (e.g. "Faculty, Accounting",
# "Faculty, Marketing"). category_source and department_source both point
# at h2 - the same heading is read twice, once as-is for the category
# check and once split for the department name. Pure category headings
# with no comma ("Administration", "Dean and Associate Deans") have
# nothing to split off, so they fall back to the faculty name. "Dean" and
# "Program Director" categories are recognized here too - verified live
# that everyone under them holds a genuine Professor/Associate Professor
# title.
_FUSED_CATEGORY_DEPARTMENT_CONFIG = {
    "category_source": "h2",
    "category_keywords": ["faculty", "dean", "program director"],
    "department_source": "h2",
    "department_parse": "split_on_comma",
    "department_fallback": "faculty_name",
}

# Confirmed working faculty-and-staff.html hub pages (verified live). Not
# every faculty listed in the academic calendar has one - e.g. School of
# International Policy and Governance returns a 404 - so this scraper is
# tolerant of missing/failed pages rather than assuming full coverage.
FACULTY_HUB_URLS = [
    ("Faculty of Arts", "https://www.wlu.ca/academics/faculties/faculty-of-arts/faculty-and-staff.html", _STANDARD_CONFIG_WITH_LEADERSHIP_CATEGORIES),
    ("Faculty of Education", "https://www.wlu.ca/academics/faculties/faculty-of-education/faculty-and-staff.html", _NO_DEPARTMENT_BREAKDOWN_CONFIG),
    ("Faculty of Human and Social Sciences", "https://www.wlu.ca/academics/faculties/faculty-of-human-and-social-sciences/faculty-and-staff.html", _STANDARD_CONFIG),
    ("Faculty of Liberal Arts", "https://www.wlu.ca/academics/faculties/faculty-of-liberal-arts/faculty-and-staff.html", _STANDARD_CONFIG),
    ("Faculty of Music", "https://www.wlu.ca/academics/faculties/faculty-of-music/faculty-and-staff.html", _CATEGORY_H2_DEPARTMENT_H3_CONFIG),
    ("Faculty of Science", "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-and-staff.html", _STANDARD_CONFIG),
    ("Faculty of Social Work", "https://www.wlu.ca/academics/faculties/faculty-of-social-work/faculty-and-staff.html", _STANDARD_CONFIG),
    ("Lazaridis School of Business and Economics", "https://www.wlu.ca/academics/faculties/lazaridis-school-of-business-and-economics/faculty-and-staff.html", _FUSED_CATEGORY_DEPARTMENT_CONFIG),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def _resolve_heading_text(source, h2_text, h3_text, faculty_name):

    if source == "h2":
        return h2_text

    if source == "h3":
        return h3_text

    if source == "faculty_name":
        return faculty_name

    return None


def _passes_category_check(config, h2_text, h3_text):

    if config["category_source"] == "none":
        return True

    category_text = _resolve_heading_text(
        config["category_source"], h2_text, h3_text, None
    )

    if not category_text:
        return False

    category_text_lower = category_text.lower()

    return any(
        keyword in category_text_lower
        for keyword in config["category_keywords"]
    )


def _resolve_department(config, h2_text, h3_text, faculty_name):

    if config["department_source"] == "faculty_name":
        return faculty_name

    raw_text = _resolve_heading_text(
        config["department_source"], h2_text, h3_text, faculty_name
    )

    department = None

    if raw_text:

        if config["department_parse"] == "split_on_comma":
            if "," in raw_text:
                department = raw_text.split(",", 1)[1].strip()

        else:  # "as_is"
            department = raw_text

    if department:
        return department

    if config["department_fallback"] == "faculty_name":
        return faculty_name

    return None


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

    for faculty_name, hub_url, config in FACULTY_HUB_URLS:

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

        current_h2 = None
        current_h3 = None
        found = 0

        for el in soup.find_all(["h2", "h3", "a"]):

            if el.name == "h2":
                current_h2 = el.get_text(strip=True)

            elif el.name == "h3":
                current_h3 = el.get_text(strip=True)

            elif el.name == "a":

                href = el.get("href")

                if not href or "faculty-profiles" not in href:
                    continue

                if not _passes_category_check(config, current_h2, current_h3):
                    continue

                department = _resolve_department(
                    config, current_h2, current_h3, faculty_name
                )

                if not department:
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

                if department not in entry["department_names"]:
                    entry["department_names"].append(department)

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
