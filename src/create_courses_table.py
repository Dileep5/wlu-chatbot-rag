import sqlite3

conn = sqlite3.connect("data/courses.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS courses")

cursor.execute("""
CREATE TABLE courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,
    course_name TEXT,
    credits TEXT,
    description TEXT,
    source_url TEXT,
    faculty_name TEXT,
    department_name TEXT
)
""")

conn.commit()
conn.close()

print("Courses table created successfully!")