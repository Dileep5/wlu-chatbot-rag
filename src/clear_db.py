import sqlite3

conn = sqlite3.connect("data/courses.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM courses")

conn.commit()
conn.close()

print("Table cleared")