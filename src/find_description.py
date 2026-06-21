
import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/course.php?c=81024&cal=3&d=3327&s=1186&y=94"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

h1 = soup.find_all("h1")[1]

print("H1:")
print(h1.get_text(" ", strip=True))

print("\nNEXT 10 ELEMENTS:\n")

current = h1

for i in range(10):
    current = current.find_next()

    if current:
        print(f"\nELEMENT {i+1}")
        print("TAG:", current.name)
        print(current.get_text(" ", strip=True)[:500])