import sqlite3

conn = sqlite3.connect("data/courses.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM courses")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()