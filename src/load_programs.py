import csv
import re
import sqlite3
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

_COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")


def _extract_required_course_refs(program_soup):

    # Graduate-only, simplified extraction (Sprint 7C/7D): structurally
    # locate the "Program Requirements" section the same way course
    # prerequisites were located in Sprint 6F (an anchored div.reqs
    # block, not a fragile phrase-count heuristic - that heuristic is
    # exactly what silently fails on undergraduate pages, confirmed
    # during the Sprint 7C investigation). Every embedded course
    # hyperlink found inside it is treated as "required" - electives,
    # categorical rules ("1.0 senior CP elective" has no hyperlink at
    # all, so it's naturally excluded), and option-specific breakdowns
    # (Thesis vs. Co-op vs. Coursework) are deliberately not
    # distinguished, per the approved simplified design.
    anchor = program_soup.find("a", attrs={"name": "Program_Requirements"})

    if not anchor:
        return []

    reqs_div = anchor.find_parent("div", class_="reqs")

    if not reqs_div:
        return []

    refs = {}

    for link in reqs_div.find_all("a", href=True):

        if "course.php" not in link["href"]:
            continue

        match = _COURSE_CODE_PATTERN.search(link.get_text(strip=True).upper())

        if not match:
            continue

        course_code = match.group()

        if course_code in refs:
            continue

        context = link.find_parent("li") or link.find_parent("p")

        raw_text = (
            context.get_text(" ", strip=True)[:300] if context else ""
        )

        refs[course_code] = raw_text

    return list(refs.items())


def load_programs(programs_csv, level):

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    # Clear only this level's old data, so loading one level never
    # wipes rows belonging to the other level.
    cursor.execute("DELETE FROM programs WHERE level = ?", (level,))

    # program_course_requirements is graduate-only by design (Sprint 7D)
    # - it never holds undergraduate rows at all, so a full clear here is
    # safe and must happen before the DELETE above would make a
    # program-name-scoped clear query return nothing (the graduate rows
    # it would need to match against no longer exist in `programs` by
    # that point).
    if level == "graduate":
        cursor.execute("DELETE FROM program_course_requirements")

    with open(
        programs_csv,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            program_name = row["program_name"]
            search_url = row["program_url"]

            print(f"\nProcessing: {program_name}")

            try:

                # ----------------------------------
                # Step 1: Open search result page
                # ----------------------------------

                response = requests.get(search_url)

                if response.status_code != 200:
                    print("Failed search page")
                    continue

                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )

                program_url = None

                # Find actual program.php link - prefer an exact text match.
                for link in soup.find_all("a"):

                    text = link.get_text(strip=True)
                    href = link.get("href")

                    if (
                        text == program_name
                        and href
                        and "program.php" in href
                    ):
                        program_url = (
                            "https://academic-calendar.wlu.ca/"
                            + href
                        )
                        break

                # Some calendars (notably undergraduate) only list a program
                # as part of combined/joint-degree links (e.g. "Honours BSc
                # in Computer Science and Honours Bachelor of Business
                # Administration"), so no link ever matches exactly. Fall
                # back to the shortest program.php link whose text contains
                # the program name as a substring - the shortest match is
                # the one closest to the standalone program rather than a
                # longer compound listing.
                if not program_url:

                    candidates = []

                    for link in soup.find_all("a"):

                        text = link.get_text(strip=True)
                        href = link.get("href")

                        if (
                            href
                            and "program.php" in href
                            and program_name in text
                        ):
                            candidates.append((text, href))

                    if candidates:
                        candidates.sort(key=lambda c: len(c[0]))
                        program_url = (
                            "https://academic-calendar.wlu.ca/"
                            + candidates[0][1]
                        )

                if not program_url:
                    print("No program page found")
                    continue

                # ----------------------------------
                # Step 2: Open program page
                # ----------------------------------

                page = requests.get(program_url)

                if page.status_code != 200:
                    print("Failed program page")
                    continue

                program_soup = BeautifulSoup(
                    page.text,
                    "html.parser"
                )

                page_text = program_soup.get_text(
                    "\n",
                    strip=True
                )

                # ----------------------------------
                # Description
                # ----------------------------------

                description = ""

                idx = page_text.find(program_name)

                if idx != -1:
                    description = page_text[
                        idx:idx + 2000
                    ]

                # ----------------------------------
                # Admission Requirements
                # ----------------------------------

                admission = ""

                if page_text.count("Admission Requirements") >= 2:

                    first = page_text.find("Admission Requirements")

                    start = page_text.find(
                        "Admission Requirements",
                        first + 1
                    )

                    end = page_text.find(
                        "Co-operative Education",
                        start
                    )

                    if end == -1:
                        end = start + 5000

                    admission = page_text[start:end]


                # ----------------------------------
                # Program Requirements
                # ----------------------------------

                requirements = ""

                if page_text.count("Program Requirements") >= 2:

                    first = page_text.find("Program Requirements")

                    start = page_text.find(
                        "Program Requirements",
                        first + 1
                    )

                    requirements = page_text[
                        start:start + 8000
                    ]

                # ----------------------------------
                # Save
                # ----------------------------------

                cursor.execute("""
                INSERT INTO programs
                (
                    program_name,
                    description,
                    admission_requirements,
                    program_requirements,
                    source_url,
                    level
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    description,
                    admission,
                    requirements,
                    program_url,
                    level
                ))

                print("Inserted")

                if level == "graduate":

                    for course_code, raw_text in _extract_required_course_refs(
                        program_soup
                    ):

                        cursor.execute("""
                        INSERT OR IGNORE INTO program_course_requirements
                        (program_name, course_code, requirement_type, raw_text)
                        VALUES (?, ?, ?, ?)
                        """, (
                            program_name,
                            course_code,
                            "required",
                            raw_text,
                        ))

            except Exception as e:

                print("Error:", e)

    conn.commit()
    conn.close()

    print("\nDone!")


# Deterministic, keyword-based classification of a program's TYPE from
# its own title (Sprint 11B) - no LLM involved, consistent with this
# project's routing philosophy. Checked in this priority order because
# several keywords can co-occur (e.g. "...in Combination with...
# Professional Experience Program Option" contains both "Combination"
# and "Option" - the combined-degree fact is the more structurally
# significant one, so it's checked first, before the option check).
# Two "Honours" mentions is a second, independent signal for a genuine
# two-DEGREE join ("Honours BSc in Computer Science and Honours
# Bachelor of Business Administration") that the word "combin*" alone
# wouldn't catch on its own, distinguishing it from a combined-SUBJECT
# single degree ("Honours BSc Computer Science and Mathematics", one
# "Honours" mention, classified as a major) - confirmed against all 14
# real Computer Science program variants during the Sprint 11A/11B
# investigation.
_PROGRAM_TYPE_RULES = [
    ("minor", re.compile(r"\bminor\b", re.IGNORECASE)),
    ("certificate", re.compile(r"\bcertificate\b", re.IGNORECASE)),
    ("concentration", re.compile(r"\b(?:concentrations?|streams?)\b", re.IGNORECASE)),
    ("combined", re.compile(r"\bcombin(?:ed|ation)\b", re.IGNORECASE)),
    ("option", re.compile(r"\boption\b", re.IGNORECASE)),
]


def _classify_program_type(program_name):

    if program_name.lower().count("honours") >= 2:
        return "combined"

    for program_type, pattern in _PROGRAM_TYPE_RULES:

        if pattern.search(program_name):
            return program_type

    return "major"


# Sprint 11C: CS-style undergraduate program pages ("Honours BSc
# Computer Science" and similar) organize required courses under
# <b>Year N</b> markers, with course codes as course.php hyperlinks
# interspersed in prose - confirmed live during Sprint 11A/11C
# investigation, structurally distinct from the two other undergraduate
# shapes found (History-style credit/category prose with no course
# links inside its own "Year N" sections, and joint-program HTML
# tables) - this sprint deliberately handles only this one shape, per
# the explicit scope. A "recommended" course mentioned as an elective
# suggestion (e.g. "2.5 elective credits - HI132 recommended") is
# excluded - it's advice, not a requirement, and treating it as one
# would be a real fabrication risk (confirmed live: HI132 appears
# hyperlinked in CP312's own Year 1 section this exact way).
_YEAR_MARKER_TEXT_PATTERN = re.compile(r"^Year\s+(\d+)", re.IGNORECASE)
_TERM_TEXT_PATTERN = re.compile(r"\b(Fall|Winter|Spring)\b", re.IGNORECASE)
_RECOMMENDED_TEXT_PATTERN = re.compile(r"\brecommended\b", re.IGNORECASE)


def _tokenize_year_content(program_soup):

    content = program_soup.find("div", class_="content")

    if not content:
        return []

    tokens = []

    for element in content.descendants:

        # "Program Regulations" and similar sections (progression GPA
        # rules, exclusions, campus notes) start here - not part of the
        # year-by-year course breakdown, and out of scope this sprint.
        if (
            isinstance(element, Tag)
            and "subanchor" in (element.get("class") or [])
        ):
            break

        if isinstance(element, Tag) and element.name == "b":

            year_match = _YEAR_MARKER_TEXT_PATTERN.match(
                element.get_text(strip=True)
            )

            if year_match:
                tokens.append(("year", int(year_match.group(1))))

            continue

        if (
            isinstance(element, Tag)
            and element.name == "a"
            and "course.php" in (element.get("href") or "")
        ):

            # Glossary-definition links (e.g. "elective") are not course
            # references at all.
            if "glossaryTerm" in (element.get("class") or []):
                continue

            code_match = _COURSE_CODE_PATTERN.search(
                element.get_text(strip=True).upper()
            )

            if code_match:
                tokens.append(("course", code_match.group()))

            continue

        if isinstance(element, NavigableString):

            # Only text not already captured structurally via its
            # parent <a>/<b> above.
            if element.parent and element.parent.name in ("a", "b"):
                continue

            text = str(element)

            if text.strip():
                tokens.append(("text", text))

    return tokens


def _extract_year_based_course_refs(program_soup):
    """Returns (is_year_marker_page, [(year, term, course_code), ...]).

    is_year_marker_page is True whenever at least one <b>Year N</b>
    marker was found at all - callers should treat this as "this page
    uses the supported CS-style shape" even when the resulting course
    list is empty (a program with Year N headings but no hyperlinked
    courses inside them, like History's credit/category style, simply
    has nothing extractable - correctly zero rows, not a fabrication;
    the caller doesn't need to tell that apart from "not this shape at
    all" since both produce the same, correct outcome: no rows stored).
    """

    tokens = _tokenize_year_content(program_soup)

    if not any(kind == "year" for kind, _ in tokens):
        return False, []

    results = []
    current_year = None
    seen_in_year = set()

    for index, (kind, value) in enumerate(tokens):

        if kind == "year":
            current_year = value
            continue

        if kind != "course" or current_year is None:
            continue

        # Look ahead through the immediately-following text only (up to
        # the next course/year token) for a term keyword and the
        # "recommended" exclusion signal - deliberately local/structural,
        # not a full parse of the surrounding sentence.
        lookahead_parts = []

        for next_kind, next_value in tokens[index + 1:]:

            if next_kind in ("course", "year"):
                break

            lookahead_parts.append(next_value)

        lookahead_text = "".join(lookahead_parts)

        if _RECOMMENDED_TEXT_PATTERN.search(lookahead_text):
            continue

        key = (current_year, value)

        if key in seen_in_year:
            continue

        seen_in_year.add(key)

        term_match = _TERM_TEXT_PATTERN.search(lookahead_text)
        term = term_match.group(1).capitalize() if term_match else None

        results.append((current_year, term, value))

    return True, results


def load_undergraduate_programs(
    program_links_csv="outputs/undergraduate_program_links.csv"
):

    # Sprint 11B: undergraduate programs discovered by
    # get_undergraduate_program_links.py already resolve directly to
    # their program.php page - unlike the graduate loader above, there
    # is no search-page redirect step needed here at all.

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    # Level-scoped, mirroring the existing pattern used for graduate
    # reloads above - never touches graduate rows. Sprint 11C puts the
    # level-scoped reload fix (Sprint 11B) to actual use for the first
    # time: this DELETE only ever touches undergraduate rows here,
    # leaving all 171 graduate program_course_requirements rows
    # completely undisturbed.
    cursor.execute("DELETE FROM programs WHERE level = 'undergraduate'")
    cursor.execute(
        "DELETE FROM program_course_requirements WHERE level = 'undergraduate'"
    )

    year_marker_pages = 0
    programs_with_requirements = 0
    requirement_rows_inserted = 0
    unsupported_pages = 0

    with open(program_links_csv, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            program_name = row["program_name"]
            program_url = row["program_url"]

            print(f"\nProcessing: {program_name}")

            try:

                response = requests.get(
                    program_url, headers=HEADERS, timeout=20
                )

                if response.status_code != 200:
                    print(f"Failed program page ({response.status_code})")
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                page_text = soup.get_text("\n", strip=True)

                # Description: same simple, already-proven capture as
                # the graduate loader above - the page's own text
                # starting from the program's own title, since
                # undergraduate pages have no structurally-distinct
                # "description" section of their own (confirmed live -
                # everything through the year-by-year requirements is
                # one continuous block of prose, and course-requirement
                # extraction is deliberately deferred past this sprint).
                description = ""

                idx = page_text.find(program_name)

                if idx != -1:
                    description = page_text[idx:idx + 2000]

                # Admission requirements: same heuristic as graduate,
                # for consistency and in case some programs do have
                # this - live investigation (Sprint 11A) found no
                # undergraduate program page with real per-program
                # admission content (only a generic sidebar navigation
                # mention), so this is expected to stay empty for most
                # or all undergraduate rows. A real absence in the
                # source data, not an extraction bug.
                admission = ""

                if page_text.count("Admission Requirements") >= 2:

                    first = page_text.find("Admission Requirements")
                    start = page_text.find("Admission Requirements", first + 1)
                    end = page_text.find("Co-operative Education", start)

                    if end == -1:
                        end = start + 5000

                    admission = page_text[start:end]

                program_type = _classify_program_type(program_name)

                # program_requirements deliberately left empty - course-
                # requirement extraction is out of scope this sprint
                # (Sprint 11C+), matching the level's absence from
                # program_course_requirements entirely.
                cursor.execute("""
                INSERT INTO programs
                (
                    program_name,
                    description,
                    admission_requirements,
                    program_requirements,
                    source_url,
                    level,
                    program_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    program_name,
                    description,
                    admission,
                    "",
                    program_url,
                    "undergraduate",
                    program_type,
                ))

                print("Inserted")

                # Sprint 11C: CS-style Year N course-requirement
                # extraction, scoped explicitly to that one shape - see
                # _extract_year_based_course_refs()'s docstring for why
                # History-style and joint-table pages both correctly
                # yield zero rows here without needing to be told apart
                # from "not this shape at all".
                is_year_marker_page, course_refs = _extract_year_based_course_refs(soup)

                if is_year_marker_page:
                    year_marker_pages += 1

                if course_refs:

                    programs_with_requirements += 1

                    for year, term, course_code in course_refs:

                        cursor.execute("""
                        INSERT OR IGNORE INTO program_course_requirements
                        (program_name, course_code, requirement_type, raw_text, level, year, term)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            program_name,
                            course_code,
                            "required",
                            None,
                            "undergraduate",
                            year,
                            term,
                        ))

                        if cursor.rowcount:
                            requirement_rows_inserted += 1

                elif not is_year_marker_page:
                    unsupported_pages += 1

            except Exception as e:

                print("Error:", e)

    conn.commit()
    conn.close()

    print("\nDone!")
    print(f"Year-marker pages detected: {year_marker_pages}")
    print(f"Programs with extracted requirements: {programs_with_requirements}")
    print(f"Requirement rows inserted: {requirement_rows_inserted}")
    print(f"Pages with no Year N markers at all (unsupported shape): {unsupported_pages}")

    return {
        "year_marker_pages": year_marker_pages,
        "programs_with_requirements": programs_with_requirements,
        "requirement_rows_inserted": requirement_rows_inserted,
        "unsupported_pages": unsupported_pages,
    }


if __name__ == "__main__":
    load_programs(
        programs_csv="outputs/graduate_programs.csv",
        level="graduate"
    )
