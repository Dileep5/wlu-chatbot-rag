import sqlite3

conn = sqlite3.connect("data/programs.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS program_course_requirements")

# Originally graduate-only, per the approved Sprint 7C simplified
# design: each row is a course explicitly hyperlinked somewhere in a
# program's "Program Requirements" section. requirement_type is always
# "required" here - electives, categorical rules ("1.0 senior CP
# elective"), and option-specific breakdowns (Thesis vs. Co-op vs.
# Coursework) are deliberately not parsed or represented; raw_text
# preserves the original surrounding text so nothing about the
# simplification is hidden from anyone reading the data later.
# UNIQUE(program_name, course_code) is what collapses a course
# mentioned in multiple options (e.g. CP600 required under both the
# Thesis and Co-op options) into a single row, consistent with "ignore
# option-specific parsing."
#
# `level` (Sprint 11B) exists so a reload of one level's programs can
# clear only that level's rows here (`DELETE ... WHERE level = ?`)
# without depending on a join back to the `programs` table - the exact
# ordering bug Sprint 7D hit once already (a subquery against a
# programs table whose matching rows had just been deleted) would
# recur immediately if undergraduate reload tried to scope its DELETE
# through such a join instead.
cursor.execute("""
CREATE TABLE program_course_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT NOT NULL,
    course_code TEXT NOT NULL,
    requirement_type TEXT NOT NULL,
    raw_text TEXT,
    level TEXT NOT NULL DEFAULT 'graduate',
    UNIQUE(program_name, course_code)
)
""")

cursor.execute("""
CREATE INDEX idx_program_course_requirements_course
ON program_course_requirements(course_code)
""")

cursor.execute("""
CREATE INDEX idx_program_course_requirements_program
ON program_course_requirements(program_name)
""")

conn.commit()
conn.close()

print("program_course_requirements table created successfully!")
