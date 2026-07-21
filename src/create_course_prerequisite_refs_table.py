import sqlite3

conn = sqlite3.connect("data/courses.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS course_prerequisite_refs")

# A derived index, not a source of truth - required_course_code values
# are extracted from courses.prerequisites_text (via regex, during
# scraping) whenever that text happens to mention a specific course
# code. Free-text prerequisites that don't reference a course code at
# all ("Permission of the instructor and the graduate studies
# committee") produce no rows here; courses.prerequisites_text remains
# the authoritative record regardless of what this table captures.
#
# level is carried alongside the reference pair purely so load_courses()
# can clear only one level's rows on reload (mirroring how `courses`
# itself is cleared per level) without wiping the other level's data -
# it isn't part of the logical identity of a reference, which is why the
# uniqueness constraint below doesn't include it.
cursor.execute("""
CREATE TABLE course_prerequisite_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL,
    required_course_code TEXT NOT NULL,
    level TEXT,
    UNIQUE(course_code, required_course_code)
)
""")

cursor.execute("""
CREATE INDEX idx_course_prerequisite_refs_required
ON course_prerequisite_refs(required_course_code)
""")

cursor.execute("""
CREATE INDEX idx_course_prerequisite_refs_course
ON course_prerequisite_refs(course_code)
""")

conn.commit()
conn.close()

print("course_prerequisite_refs table created successfully!")
