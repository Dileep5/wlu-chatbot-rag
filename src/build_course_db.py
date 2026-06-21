import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,
    title TEXT,
    credits TEXT,
    description TEXT,
    url TEXT
)
""")

conn.commit()

print("Table created successfully!")

conn.close()