
import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/course.php?c=81024&cal=3&d=3327&s=1186&y=94"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

for tag in soup.find_all():
    if "CP640" in tag.get_text():
        print("\nTAG:", tag.name)
        print(tag.get_text(" ", strip=True))
        print("-" * 100)