import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,
    course_name TEXT,
    credits TEXT,
    description TEXT,
    source_url TEXT
)
""")

conn.commit()
conn.close()

print("Courses table created successfully!")