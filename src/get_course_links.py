import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/department.php?cal=3&d=3327&s=1186&y=94"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

links = soup.find_all("a")

for link in links:
    href = link.get("href")

    if href and "course.php" in href:
        print(link.text.strip(), "->", href)