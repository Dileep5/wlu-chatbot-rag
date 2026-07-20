import csv
import sqlite3
import requests
from bs4 import BeautifulSoup


def load_programs(programs_csv, level):

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    # Clear only this level's old data, so loading one level never
    # wipes rows belonging to the other level.
    cursor.execute("DELETE FROM programs WHERE level = ?", (level,))

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
