import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/course.php?c=81018&cal=3&d=3327&s=1186&y=94"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

text = soup.get_text()

with open("course_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Saved!")