import sqlite3

conn = sqlite3.connect("data/departments.db")
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS departments
""")

cursor.execute("""
CREATE TABLE departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faculty_name TEXT,
    department_name TEXT,
    coordinator TEXT,
    programs TEXT,
    description TEXT,
    source_url TEXT,
    level TEXT
)
""")

conn.commit()
conn.close()

print("Departments table created successfully!")