import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("""
INSERT INTO courses
(course_code, title, credits, description, url)
VALUES (?, ?, ?, ?, ?)
""",
(
    "CP600",
    "Practical Algorithm Design",
    "0.5",
    "The techniques of algorithm design form one of the core practical technologies of computer science.",
    "https://academic-calendar.wlu.ca/course.php?c=81018"
))

conn.commit()

print("Course saved!")

conn.close()