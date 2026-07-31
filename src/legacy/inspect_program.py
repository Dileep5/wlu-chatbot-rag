import requests
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/program.php?cal=3&d=3327&p=7778&s=1186&y=94"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text("\n", strip=True)

start = text.find("Admission Requirements")

print(text[start:start+3000])