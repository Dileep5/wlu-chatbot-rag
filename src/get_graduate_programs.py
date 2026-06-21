import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/index_old.php?cal=3&y=94"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

for link in soup.find_all("a"):

    text = link.get_text(strip=True)
    href = link.get("href")

    if href:
        print(text, "->", href)