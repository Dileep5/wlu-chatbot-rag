import sqlite3

conn = sqlite3.connect("data/policies.db")
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS policies
""")

cursor.execute("""
CREATE TABLE policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_number TEXT UNIQUE,
    policy_title TEXT,
    source_url TEXT
)
""")

conn.commit()
conn.close()

print("Policies table created successfully!")
