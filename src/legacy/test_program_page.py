import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/search.php?cal=3&s_items=departments+programs&s_pdt=67&y=94"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

for link in soup.find_all("a"):

    text = link.get_text(strip=True)
    href = link.get("href")

    if text and href:
        print(text, "->", href)