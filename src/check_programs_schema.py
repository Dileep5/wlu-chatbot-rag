import sqlite3

conn = sqlite3.connect("data/programs.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(programs)")

for row in cursor.fetchall():
    print(row)

conn.close()