import sqlite3

conn = sqlite3.connect("data/faculty.db")
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS faculty
""")

cursor.execute("""
CREATE TABLE faculty (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    title TEXT,
    faculty_name TEXT,
    department_name TEXT,
    email TEXT,
    phone TEXT,
    office TEXT,
    research_interests TEXT,
    biography TEXT,
    source_url TEXT
)
""")

conn.commit()
conn.close()

print("Faculty table created successfully!")
