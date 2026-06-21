import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("PRAGMA table_info(courses)")

for row in cursor.fetchall():
    print(row)

conn.close()