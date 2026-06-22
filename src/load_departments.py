import csv
import sqlite3
import requests
from bs4 import BeautifulSoup

# Connect database
conn = sqlite3.connect("data/departments.db")
cursor = conn.cursor()

# Clear old data
cursor.execute("DELETE FROM departments")

# Read departments CSV
with open("outputs/departments.csv", newline="", encoding="utf-8") as file:

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

            # -------------------------
            # Coordinator
            # -------------------------

            coordinator = ""

            if "Graduate Program Co-ordinator" in page_text:
                idx = page_text.find("Graduate Program Co-ordinator")
                coordinator = page_text[idx:idx + 500]

            elif "Graduate Program Co-ordinators" in page_text:
                idx = page_text.find("Graduate Program Co-ordinators")
                coordinator = page_text[idx:idx + 500]

            # -------------------------
            # Programs
            # -------------------------

            programs = []

            for line in page_text.split("\n"):

                line = line.strip()

                if (
                    line.startswith("Master of")
                    or line.startswith("Doctor of")
                    or line.startswith("Graduate Diploma")
                ):
                    programs.append(line)

            programs_text = " | ".join(sorted(set(programs)))

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
                source_url
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                faculty_name,
                department_name,
                coordinator,
                programs_text,
                description,
                url
            ))

            print("Inserted")

        except Exception as e:

            print("Error:", e)

conn.commit()
conn.close()

print("\nDone!")