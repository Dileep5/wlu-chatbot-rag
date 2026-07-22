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

# Only a letter-group directly adjacent to digits can match - this is
# what a broader pattern (allowing a second optional letter-group
# separated by whitespace) got wrong during investigation: it let an
# unrelated stray token ("II") get absorbed into an adjacent real code
# ("HS206") as if it were one match. Cross-listed codes ("CP/PC400") are
# still supported, but only when joined by a literal "/".
_COURSE_CODE_PATTERN = re.compile(
    r"\b([A-Z]{2,4}(?:/[A-Z]{2,4})?)\s?(\d{3})([A-Z]?)\b"
)


def _extract_courses_taught_codes(text):

    # Cross-listed codes ("CP/PC400") are split into their two individual,
    # separately-registerable codes ("CP400", "PC400") rather than kept
    # as one joined string - course_code needs to be atomic to match
    # courses.db and to answer "who taught CP400" / "who taught PC400"
    # symmetrically. Deduplicated per-profile here (not left to the
    # table's UNIQUE constraint) so a code mentioned twice on one page
    # (e.g. once under "will teach", once under "recent courses") doesn't
    # depend on insert-order/ignore behavior to collapse correctly.
    codes = []
    seen = set()

    for match in _COURSE_CODE_PATTERN.finditer(text):

        prefix, digits, suffix = match.groups()

        if "/" in prefix:
            first, second = prefix.split("/", 1)
            normalized_list = [first + digits + suffix, second + digits + suffix]
        else:
            normalized_list = [prefix + digits + suffix]

        for normalized in normalized_list:

            normalized = normalized.upper().replace(" ", "")

            if normalized not in seen:
                seen.add(normalized)
                codes.append(normalized)

    return codes


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


# Any *.wlu.ca subdomain, not just the literal "wlu.ca" - confirmed live
# that some faculty (e.g. retired/emeritus accounts) use a subdomain
# like "ret.wlu.ca" that the original wlu.ca-only pattern missed.
_EMAIL_PATTERN = re.compile(
    r"[\w.\-]+@(?:[\w\-]+\.)*wlu\.ca", re.IGNORECASE
)


def _deobfuscate_email_text(text):

    # Anti-spam text obfuscation confirmed live in both visible link text
    # ("jaguinaldo [at] wlu.ca") and, in one case, inside the mailto href
    # itself ("mailto:pmallet [at] wlu [dot] ca") - normalized the same
    # way regardless of where it's found.
    text = re.sub(r"\s*\[\s*at\s*\]\s*", "@", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\[\s*dot\s*\]\s*", ".", text, flags=re.IGNORECASE)

    return text


def _extract_email(soup, text):

    # mailto hrefs first - the most reliable, structural source, and the
    # only place the address exists at all for icon-only "Email" links
    # with no visible address text. Only *.wlu.ca addresses are ever
    # returned - a personal/external address in a mailto href (e.g. a
    # Gmail account, confirmed live for one profile) is deliberately not
    # surfaced, since that's a scope decision, not an extraction bug.
    for anchor in soup.find_all("a", href=True):

        href = anchor["href"]

        if not href.lower().startswith("mailto:"):
            continue

        candidate = _deobfuscate_email_text(href[len("mailto:"):])
        match = _EMAIL_PATTERN.search(candidate)

        if match:
            return match.group()

    # Falls back to visible page text, also de-obfuscated, for any
    # address shown as plain (possibly obfuscated) text with no mailto
    # link backing it.
    match = _EMAIL_PATTERN.search(_deobfuscate_email_text(text))

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
    cursor.execute("DELETE FROM faculty_courses_taught")

    # Loaded once (not per-profile) purely to set the in_courses_db flag -
    # courses.db is a separate database file, so this is a plain Python
    # set lookup rather than a cross-database SQL join.
    courses_conn = sqlite3.connect("data/courses.db")
    courses_cursor = courses_conn.cursor()
    courses_cursor.execute("SELECT course_code FROM courses")
    known_course_codes = {
        row[0].upper().replace(" ", "") for row in courses_cursor.fetchall()
    }
    courses_conn.close()

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

                # response.text decodes using requests' guessed encoding,
                # which falls back to ISO-8859-1 whenever the server's
                # Content-Type header omits a charset - confirmed root
                # cause of the mojibake corruption in faculty names
                # (e.g. "AngÃ¨le Foley" for "Angèle Foley", Sprint 10C).
                # response.content is the raw bytes; BeautifulSoup's own
                # encoding detection correctly identifies UTF-8 instead
                # of inheriting requests' wrong guess.
                soup = BeautifulSoup(response.content, "html.parser")

                name = _extract_name(soup, member_name)
                title = _extract_title(soup)

                research_interests = _extract_accordion_section(
                    soup, ["research"]
                )

                biography = _extract_accordion_section(
                    soup, ["biography"]
                )

                # Exact heading match, not a loose "course"/"teaching"
                # keyword - investigation found a differently-headed
                # section ("Anecdotal Thoughts on Teaching at Laurier")
                # that's personal teaching-philosophy prose with no course
                # codes at all, which a looser keyword would have wrongly
                # matched.
                courses_taught_text = _extract_accordion_section(
                    soup, ["courses taught"]
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

                email = _extract_email(soup, page_text)
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

                if courses_taught_text:

                    for course_code in _extract_courses_taught_codes(
                        courses_taught_text
                    ):

                        cursor.execute("""
                        INSERT OR IGNORE INTO faculty_courses_taught
                        (
                            faculty_source_url,
                            course_code,
                            raw_text,
                            term_label,
                            in_courses_db
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """, (
                            profile_url,
                            course_code,
                            courses_taught_text,
                            None,
                            1 if course_code in known_course_codes else 0
                        ))

            except Exception as e:

                print("Error:", e)

    conn.commit()
    conn.close()

    print("\nDone!")


if __name__ == "__main__":
    load_faculty(faculty_links_csv="outputs/faculty_links.csv")
