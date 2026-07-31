import sqlite3

conn = sqlite3.connect("data/departments.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(departments)")

for row in cursor.fetchall():
    print(row)

conn.close()