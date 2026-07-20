import csv
import re
import sqlite3
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def _extract_name(soup, member_name):

    # The page's own <h1> is exactly the person's name (verified across
    # multiple faculties/departments) - a direct structural read, not a
    # guess based on where their name happens to appear in the page text.
    # A couple of real WLU profile pages have their job title authored
    # into the <h1> instead of the name (a content mistake on the site
    # itself, not a scraping issue) - guard against that by requiring the
    # h1 to share at least one word with the already-known roster name,
    # falling back to the roster name otherwise. That guard only makes
    # sense when there's an actual roster name to check against - some hub
    # pages have stray, empty-text anchor tags that produce an empty
    # member_name, and an empty roster name can never share a word with
    # anything, which would otherwise reject a perfectly valid h1.
    h1 = soup.find("h1")

    if h1:
        text = h1.get_text(strip=True)

        if text:

            if not member_name.strip():
                return text

            roster_words = set(member_name.lower().split())
            h1_words = set(text.lower().split())

            if roster_words & h1_words:
                return text

    return member_name


def _extract_title(soup):

    title_el = soup.find("p", class_="faculty_description")

    return title_el.get_text(strip=True) if title_el else ""


def _extract_accordion_section(soup, heading_keywords):

    # Each profile section is a <div class="accordion-single"> containing
    # an <h2 class="accordion-header"> (the section title) and a sibling
    # <div class="accordion-panel"> (the section body). Matching against
    # these specific, template-defined headers - not arbitrary page text -
    # is what makes this safe: there's no equivalent of "Research Strengths"
    # or a publication citation living inside this structure to collide
    # with.
    for header in soup.find_all("h2", class_="accordion-header"):

        header_text = header.get_text(strip=True).lower()

        if not any(keyword in header_text for keyword in heading_keywords):
            continue

        wrapper = header.parent

        if not wrapper:
            continue

        panel = wrapper.find("div", class_="accordion-panel")

        if panel:
            return panel.get_text(" ", strip=True)

    return ""


def _extract_email(text):

    match = re.search(r"[\w\.-]+@wlu\.ca", text, re.IGNORECASE)

    return match.group() if match else ""


def _extract_phone(text):

    match = re.search(r"\d{3}[.\-]\d{3}[.\-]\d{4}(?:\s*x\d+)?", text)

    return match.group() if match else ""


def _extract_office(lines):

    for line in lines:

        if line.lower().startswith("office location:"):
            return line.split(":", 1)[1].strip()

    return ""


def load_faculty(faculty_links_csv):

    conn = sqlite3.connect("data/faculty.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM faculty")

    with open(faculty_links_csv, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            faculty_name = row["faculty_name"]
            department_name = row["department_name"]
            member_name = row["member_name"]
            profile_url = row["profile_url"]

            print(f"\nProcessing: {member_name}")

            try:

                response = requests.get(
                    profile_url, headers=HEADERS, timeout=20
                )

                if response.status_code != 200:
                    print("Failed profile page")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                name = _extract_name(soup, member_name)
                title = _extract_title(soup)

                research_interests = _extract_accordion_section(
                    soup, ["research"]
                )

                biography = _extract_accordion_section(
                    soup, ["biography"]
                )

                # email/phone/office aren't part of the accordion sections,
                # so these still read from the cleaned page text.
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                if soup.head:
                    soup.head.decompose()

                page_text = soup.get_text("\n", strip=True)

                lines = [
                    line.strip()
                    for line in page_text.split("\n")
                    if line.strip()
                ]

                email = _extract_email(page_text)
                phone = _extract_phone(page_text)
                office = _extract_office(lines)

                cursor.execute("""
                INSERT INTO faculty
                (
                    name,
                    title,
                    faculty_name,
                    department_name,
                    email,
                    phone,
                    office,
                    research_interests,
                    biography,
                    source_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name,
                    title,
                    faculty_name,
                    department_name,
                    email,
                    phone,
                    office,
                    research_interests,
                    biography,
                    profile_url
                ))

                print("Inserted")

            except Exception as e:

                print("Error:", e)

    conn.commit()
    conn.close()

    print("\nDone!")


if __name__ == "__main__":
    load_faculty(faculty_links_csv="outputs/faculty_links.csv")
