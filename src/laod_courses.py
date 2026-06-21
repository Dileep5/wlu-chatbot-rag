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
        course_links.append(full_url)

print(f"Found {len(course_links)} course links")

# Visit each course page
for url in course_links:

    page = requests.get(url)
    course_soup = BeautifulSoup(page.text, "html.parser")

    try:
        # Second h1 contains course info
        header = course_soup.find_all("h1")[1].get_text(" ", strip=True)

        # Extract course code
        course_code = re.search(r"CP\d{3}", header).group()

        # Extract credits
        credits = re.search(r"(\d+\.\d+)\s*Credit", header).group(1)

        # Extract title
        title = header
        title = re.sub(r"CP\d{3}", "", title)
        title = re.sub(r"\d+\.\d+\s*Credit", "", title)
        title = title.strip()

        # Extract description
        h1 = course_soup.find_all("h1")[1]

        p_tag = h1.find_next("p")

        if p_tag:
            description = p_tag.get_text(" ", strip=True)
        else:
            description = ""

        # Insert into database
        cursor.execute("""
        INSERT INTO courses
        (course_code, title, credits, description, url)
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