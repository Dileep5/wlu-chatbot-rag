import csv
import sqlite3
import requests
from bs4 import BeautifulSoup

# Program-name line prefixes and the "who runs this department" text marker
# differ by calendar level (verified against real graduate and undergraduate
# department pages).
GRADUATE_PROGRAM_PREFIXES = [
    "Master of",
    "Doctor of",
    "Graduate Diploma",
]

UNDERGRADUATE_PROGRAM_PREFIXES = [
    "Honours Bachelor of",
    "Honours BA",
    "Honours BSc",
    "Bachelor of",
    "General BA",
    "General BSc",
    "Diploma in",
    "Minor in",
    "Specialization in",
]


def extract_coordinator(page_text, level):

    if level == "graduate":

        for marker in (
            "Graduate Program Co-ordinator",
            "Graduate Program Co-ordinators",
        ):
            if marker in page_text:
                idx = page_text.find(marker)
                return page_text[idx:idx + 500]

        return ""

    # undergraduate: pages use a "[Chair]" tag placed right after the
    # person's name, e.g. "Matthew Smith [Chair]" - capture a window
    # ending at the marker so the preceding name is included.
    marker = "[Chair]"

    if marker in page_text:
        idx = page_text.find(marker)
        start = max(0, idx - 120)
        end = idx + len(marker)
        return page_text[start:end]

    return ""


def extract_programs(page_text, level):

    prefixes = (
        GRADUATE_PROGRAM_PREFIXES
        if level == "graduate"
        else UNDERGRADUATE_PROGRAM_PREFIXES
    )

    programs = []

    for line in page_text.split("\n"):

        line = line.strip()

        if any(line.startswith(prefix) for prefix in prefixes):
            programs.append(line)

    return " | ".join(sorted(set(programs)))


def load_departments(departments_csv, level):

    conn = sqlite3.connect("data/departments.db")
    cursor = conn.cursor()

    # Clear only this level's old data, so loading one level never
    # wipes rows belonging to the other level.
    cursor.execute("DELETE FROM departments WHERE level = ?", (level,))

    with open(departments_csv, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            faculty_name = row["faculty_name"]
            department_name = row["department_name"]
            url = row["department_url"]

            print(f"\nScanning: {department_name}")

            try:

                response = requests.get(url)

                if response.status_code != 200:
                    print("Failed")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                page_text = soup.get_text("\n", strip=True)

                coordinator = extract_coordinator(page_text, level)

                programs_text = extract_programs(page_text, level)

                # -------------------------
                # Description
                # -------------------------

                description = ""

                lines = page_text.split("\n")

                for i, line in enumerate(lines):

                    if line.strip() == department_name:

                        description = "\n".join(
                            lines[i + 1:i + 20]
                        )

                        break

                # Insert
                cursor.execute("""
                INSERT INTO departments
                (
                    faculty_name,
                    department_name,
                    coordinator,
                    programs,
                    description,
                    source_url,
                    level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    faculty_name,
                    department_name,
                    coordinator,
                    programs_text,
                    description,
                    url,
                    level
                ))

                print("Inserted")

            except Exception as e:

                print("Error:", e)

    conn.commit()
    conn.close()

    print("\nDone!")


if __name__ == "__main__":
    load_departments(
        departments_csv="outputs/departments.csv",
        level="graduate"
    )
