import requests
import sqlite3
import re
import csv
from bs4 import BeautifulSoup

# Course-code shape reused everywhere else in this codebase - kept
# identical so a reference extracted here matches what search_course()
# and friends already look for.
_COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")


def _normalize_whitespace(text):

    return re.sub(r"\s+", " ", text).strip()


def _clean_requirement_text(dd_tag):

    # Space-joined, not newline-joined: this content frequently contains
    # inline hyperlinks to specific referenced course codes (e.g. "CP264
    # or (CP114 and CP213)"), and get_text("\n", ...) - the separator
    # used elsewhere in this codebase for prose sections - would
    # fragment an otherwise readable sentence with a stray newline
    # around every linked code.
    text = dd_tag.get_text(" ", strip=True)
    text = _normalize_whitespace(text)

    # Cosmetic artifact of space-joining around inline links/punctuation
    # (e.g. "CP213 )" instead of "CP213)").
    text = re.sub(r"\s+([),.;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)

    return text


def _extract_additional_course_info(soup):

    info = {
        "prerequisites_text": "",
        "exclusions_text": "",
        "corequisites_text": "",
        "notes_text": "",
        "location_text": "",
    }

    reqs_div = soup.find("div", class_="reqs")

    if not reqs_div:
        return info

    dl = reqs_div.find("dl")

    if not dl:
        return info

    for dt in dl.find_all("dt"):

        dd = dt.find_next_sibling("dd")

        if not dd:
            continue

        text = _clean_requirement_text(dd)

        if not text:
            continue

        # Substring match on a hyphen/space-stripped label, not an exact
        # match - confirmed live that a small number of course pages
        # fuse two labels into one string with no separator at all
        # ("Co-requisitesorPrerequisites"), which an exact match would
        # miss entirely. That fused text genuinely describes both
        # requirement types at once, so setting both fields isn't a
        # mismatch - it's what the source actually says.
        label_key = re.sub(r"[\s\-]", "", dt.get_text(strip=True).lower())

        if "prerequisite" in label_key:
            info["prerequisites_text"] = text

        if "corequisite" in label_key:
            info["corequisites_text"] = text

        if "exclusion" in label_key:
            info["exclusions_text"] = text

        if "note" in label_key:
            info["notes_text"] = text

        if "location" in label_key:
            info["location_text"] = text

    return info


def _extract_prerequisite_refs(course_code, prerequisites_text):

    if not prerequisites_text:
        return []

    codes = set(_COURSE_CODE_PATTERN.findall(prerequisites_text.upper()))
    codes.discard(course_code.upper())

    return sorted(codes)


def load_courses(course_links_csv, level):

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    # Clear only this level's old data, so loading one level never
    # wipes rows belonging to the other level (also prevents duplicate
    # rows if this is re-run for the same level).
    cursor.execute("DELETE FROM courses WHERE level = ?", (level,))
    cursor.execute(
        "DELETE FROM course_prerequisite_refs WHERE level = ?", (level,)
    )

    course_links = []

    with open(
        course_links_csv,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            course_links.append({
                "faculty_name": row["faculty_name"],
                "department_name": row["department_name"],
                "course_url": row["course_url"]
            })

    print(f"Found {len(course_links)} course links")

    # Visit each course page
    for course in course_links:

        faculty_name = course["faculty_name"]
        department_name = course["department_name"]
        url = course["course_url"]

        try:

            page = requests.get(url)

            if page.status_code != 200:
                print(f"Failed: {url}")
                continue

            course_soup = BeautifulSoup(page.text, "html.parser")

            # Second h1 contains course information
            h1_tags = course_soup.find_all("h1")

            if len(h1_tags) < 2:
                print(f"No course header found: {url}")
                continue

            header = h1_tags[1].get_text(" ", strip=True)

            # Examples:
            # CP640 Machine Learning 0.5 Credit
            # HI600A The Nature and Practice of History 0.5 Credit

            code_match = re.search(
                r"([A-Z]{2,4}\d{3}[A-Z]?)",
                header
            )

            credit_match = re.search(
                r"(\d+\.\d+)\s*Credits?",
                header,
                re.IGNORECASE
            )

            if not code_match:
                print(f"Could not extract course code: {url}")
                continue

            course_code = code_match.group(1)

            credits = ""

            if credit_match:
                credits = credit_match.group(1)

            # Remove course code
            title = re.sub(
                r"^[A-Z]{2,4}\d{3}[A-Z]?\s*",
                "",
                header
            )

            # Remove credits
            title = re.sub(
                r"\s*\d+\.\d+\s*Credits?$",
                "",
                title,
                flags=re.IGNORECASE
            )

            title = title.strip()

            # Description = first paragraph after h1
            description = ""

            p_tag = h1_tags[1].find_next("p")

            if p_tag:
                description = p_tag.get_text(
                    " ",
                    strip=True
                )

            additional_info = _extract_additional_course_info(course_soup)

            cursor.execute("""
            INSERT INTO courses
            (
                course_code,
                course_name,
                credits,
                description,
                source_url,
                faculty_name,
                department_name,
                level,
                prerequisites_text,
                exclusions_text,
                corequisites_text,
                notes_text,
                location_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course_code,
                title,
                credits,
                description,
                url,
                faculty_name,
                department_name,
                level,
                additional_info["prerequisites_text"],
                additional_info["exclusions_text"],
                additional_info["corequisites_text"],
                additional_info["notes_text"],
                additional_info["location_text"],
            ))

            for required_code in _extract_prerequisite_refs(
                course_code, additional_info["prerequisites_text"]
            ):

                cursor.execute("""
                INSERT OR IGNORE INTO course_prerequisite_refs
                (course_code, required_course_code, level)
                VALUES (?, ?, ?)
                """, (course_code, required_code, level))

            print(
                f"Inserted: {course_code} | {department_name}"
            )

        except Exception as e:

            print(f"Error processing: {url}")
            print(e)

    conn.commit()
    conn.close()

    print("Done!")


if __name__ == "__main__":
    load_courses(
        course_links_csv="outputs/course_links.csv",
        level="graduate"
    )
