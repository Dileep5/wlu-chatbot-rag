import requests
import csv
from bs4 import BeautifulSoup


def scrape_programs(cal, year, output_path):

    url = f"https://academic-calendar.wlu.ca/index_old.php?cal={cal}&y={year}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    programs = []

    for link in soup.find_all("a"):

        text = link.get_text(strip=True)
        href = link.get("href")

        if href and "s_pdt=" in href:

            full_url = "https://academic-calendar.wlu.ca/" + href

            programs.append([text, full_url])

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["program_name", "program_url"])

        writer.writerows(programs)

    print(f"Saved {len(programs)} programs")

    return programs


if __name__ == "__main__":
    scrape_programs(cal=3, year=94, output_path="outputs/graduate_programs.csv")
