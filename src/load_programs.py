import csv
import re
import sqlite3
import requests
from bs4 import BeautifulSoup

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


if __name__ == "__main__":
    load_programs(
        programs_csv="outputs/graduate_programs.csv",
        level="graduate"
    )
