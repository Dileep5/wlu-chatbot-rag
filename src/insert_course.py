import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("""
INSERT INTO courses
(course_code, title, credits, description, url)
VALUES (?, ?, ?, ?, ?)
""",
(
    "CP640",
    "Machine Learning",
    "0.5",
    "Machine learning is the science of getting computers to act without being explicitly programmed.",
    "https://academic-calendar.wlu.ca/course.php?c=81024"
))

conn.commit()

print("Course inserted successfully!")

conn.close()