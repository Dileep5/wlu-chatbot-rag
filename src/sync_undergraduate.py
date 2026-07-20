from get_faculties import scrape_faculties
from save_departments import scrape_departments
from save_graduate_programs import scrape_programs
from get_all_course_links import scrape_course_links
from load_departments import load_departments
from load_programs import load_programs
from laod_courses import load_courses

# Current, publicly published undergraduate calendar
# (verified at https://academic-calendar.wlu.ca/index.php).
CAL = 1
YEAR = 92
LEVEL = "undergraduate"

FACULTIES_CSV = "outputs/undergraduate_faculties.csv"
DEPARTMENTS_CSV = "outputs/undergraduate_departments.csv"
PROGRAMS_CSV = "outputs/undergraduate_programs.csv"
COURSE_LINKS_CSV = "outputs/undergraduate_course_links.csv"


def main():

    print("=" * 60)
    print("STEP 1/6: Scraping undergraduate faculties")
    print("=" * 60)
    scrape_faculties(cal=CAL, year=YEAR, output_path=FACULTIES_CSV)

    print("\n" + "=" * 60)
    print("STEP 2/6: Scraping undergraduate departments")
    print("=" * 60)
    scrape_departments(
        faculties_csv=FACULTIES_CSV,
        output_path=DEPARTMENTS_CSV
    )

    print("\n" + "=" * 60)
    print("STEP 3/6: Scraping undergraduate programs")
    print("=" * 60)
    scrape_programs(cal=CAL, year=YEAR, output_path=PROGRAMS_CSV)

    print("\n" + "=" * 60)
    print("STEP 4/6: Scraping undergraduate course links")
    print("=" * 60)
    scrape_course_links(
        departments_csv=DEPARTMENTS_CSV,
        output_path=COURSE_LINKS_CSV
    )

    print("\n" + "=" * 60)
    print("STEP 5/6: Loading undergraduate departments and programs")
    print("=" * 60)
    load_departments(departments_csv=DEPARTMENTS_CSV, level=LEVEL)
    load_programs(programs_csv=PROGRAMS_CSV, level=LEVEL)

    print("\n" + "=" * 60)
    print("STEP 6/6: Loading undergraduate courses")
    print("=" * 60)
    load_courses(course_links_csv=COURSE_LINKS_CSV, level=LEVEL)

    print("\nUndergraduate calendar sync complete.")


if __name__ == "__main__":
    main()
