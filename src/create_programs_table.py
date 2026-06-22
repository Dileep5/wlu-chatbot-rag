import sqlite3

conn = sqlite3.connect("data/programs.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS programs")

cursor.execute("""
CREATE TABLE programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT,
    description TEXT,
    admission_requirements TEXT,
    program_requirements TEXT,
    source_url TEXT
)
""")

conn.commit()
conn.close()

print("Programs table created successfully!")