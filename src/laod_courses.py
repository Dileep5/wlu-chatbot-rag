import requests
import sqlite3
import re
import csv
from bs4 import BeautifulSoup

# Connect to database
conn = sqlite3.connect("data/courses.db")
cursor = conn.cursor()

# Read all course links from CSV
course_links = []

with open(
    "outputs/course_links.csv",
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

        cursor.execute("""
        INSERT INTO courses
        (
            course_code,
            course_name,
            credits,
            description,
            source_url,
            faculty_name,
            department_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            course_code,
            title,
            credits,
            description,
            url,
            faculty_name,
            department_name
        ))

        print(
            f"Inserted: {course_code} | {department_name}"
        )

    except Exception as e:

        print(f"Error processing: {url}")
        print(e)

conn.commit()
conn.close()

print("Done!")