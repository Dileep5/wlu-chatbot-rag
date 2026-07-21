import sqlite3

conn = sqlite3.connect("data/faculty.db")
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS faculty_courses_taught
""")

# faculty_source_url (not faculty.id) is the link back to the faculty
# table. faculty.id is an AUTOINCREMENT key that load_faculty.py
# reassigns on every full rebuild (DELETE + re-insert), so a raw
# faculty_id foreign key here would silently go stale the next time
# faculty.db is reloaded. source_url is stable across rebuilds - it's
# derived from the actual profile URL, not a row-insertion counter - so
# it survives reloads of either table independently.
cursor.execute("""
CREATE TABLE faculty_courses_taught (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    faculty_source_url TEXT NOT NULL,
    course_code TEXT NOT NULL,
    raw_text TEXT,
    term_label TEXT,
    in_courses_db INTEGER NOT NULL,
    UNIQUE(faculty_source_url, course_code)
)
""")

cursor.execute("""
CREATE INDEX idx_faculty_courses_taught_code
ON faculty_courses_taught(course_code)
""")

conn.commit()
conn.close()

print("faculty_courses_taught table created successfully!")
