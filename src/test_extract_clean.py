import requests
import re
from bs4 import BeautifulSoup

url = "https://academic-calendar.wlu.ca/course.php?c=81024&cal=3&d=3327&s=1186&y=94"

html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# Second h1 contains course info
header = soup.find_all("h1")[1].get_text(" ", strip=True)

print("HEADER:")
print(header)

# Extract course code
course_code = re.search(r'CP\d{3}', header).group()

# Extract credits
credits = re.search(r'(\d+\.\d+)\s*Credit', header).group(1)

# Extract title
title = header
title = re.sub(r'CP\d{3}', '', title)
title = re.sub(r'\d+\.\d+\s*Credit', '', title)
title = title.strip()

print("\nCourse Code:", course_code)
print("Title:", title)
print("Credits:", credits)

# Find course description

all_divs = soup.find_all("div")

for div in all_divs:
    text = div.get_text(" ", strip=True)

    if course_code in text and title in text and "Credit" in text:

        print("\nCOURSE BLOCK:\n")
        print(text[:1500])
        break