import requests
import sqlite3
import re
from bs4 import BeautifulSoup

# Connect to database
conn = sqlite3.connect("data/courses.db")
cursor = conn.cursor()

# Physics & Computer Science page
department_url = "https://academic-calendar.wlu.ca/department.php?cal=3&d=3327&s=1186&y=94"

response = requests.get(department_url)
soup = BeautifulSoup(response.text, "html.parser")

course_links = []

# Find all course links
for link in soup.find_all("a"):

    href = link.get("href")

    if href and "course.php" in href:

        full_url = "https://academic-calendar.wlu.ca/" + href

        if full_url not in course_links:
            course_links.append(full_url)

print(f"Found {len(course_links)} course links")

# Visit each course page
for url in course_links:

    try:

        page = requests.get(url)

        if page.status_code != 200:
            print(f"Failed: {url}")
            continue

        course_soup = BeautifulSoup(page.text, "html.parser")

        # Second h1 contains clean course info
        h1_tags = course_soup.find_all("h1")

        if len(h1_tags) < 2:
            print(f"No course header found: {url}")
            continue

        header = h1_tags[1].get_text(" ", strip=True)

        # Example:
        # CP640 Machine Learning 0.5 Credit

        code_match = re.search(r"([A-Z]{2,4}\d{3}[A-Z]?)", header)
        credit_match = re.search(r"(\d+\.\d+)\s*Credit", header)

        if not code_match:
            print(f"Could not extract course code: {url}")
            continue

        course_code = code_match.group(1)

        credits = ""
        if credit_match:
            credits = credit_match.group(1)

        # Remove code and credits from header
        title = header

        title = re.sub(
            r"^[A-Z]{2,4}\d{3}[A-Z]?\s*",
            "",
            title
        )

        title = re.sub(
            r"\s*\d+\.\d+\s*Credits?$",
            "",
            title
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
        (course_code, course_name, credits, description, source_url)
        VALUES (?, ?, ?, ?, ?)
        """, (
            course_code,
            title,
            credits,
            description,
            url
    ))

        print(f"Inserted: {course_code} - {title}")

    except Exception as e:

        print(f"Error processing {url}")
        print(e)

conn.commit()
conn.close()

print("Done!")