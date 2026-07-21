import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from streamlit.testing.v1 import AppTest

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
APP_PATH = str(SRC_DIR / "app.py")

sys.path.insert(0, str(SRC_DIR))


# -----------------------------
# Test case definition
# -----------------------------

@dataclass
class TestCase:
    name: str
    turns: List[str]          # user messages sent in order; only the last is "the question"
    check: Callable[[str, AppTest], bool]
    category: str


# -----------------------------
# Response helpers
# -----------------------------

def contains_any(text: str, *keywords: str) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def contains_all(text: str, *keywords: str) -> bool:
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def is_non_empty_response(text: str) -> bool:
    return bool(text.strip()) and "bot error" not in text.lower()


def is_off_topic_decline(text: str) -> bool:
    lower = text.lower()
    return "wilfrid laurier" in lower and "only" in lower


# -----------------------------
# Chatbot driver (via Streamlit AppTest - runs the real app.py unmodified)
# -----------------------------

def get_last_response_text(at: AppTest) -> str:
    last = at.chat_message[-1]

    if len(last.markdown) > 0:
        return last.markdown[0].value

    # Bot hit its except-block and rendered via st.error() this run.
    try:
        if len(last.error) > 0:
            return last.error[0].value
    except AttributeError:
        pass

    return ""


def run_turns(turns: List[str]) -> str:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=90)

    response = ""

    for turn in turns:
        at.chat_input[0].set_value(turn).run(timeout=90)
        response = get_last_response_text(at)

    return response


# -----------------------------
# Test cases, grouped by category.
# Add new cases by appending a TestCase(...) to the relevant list.
# -----------------------------

BASIC_CONVERSATION_TESTS = [
    TestCase(
        name="Greeting - hi",
        turns=["hi"],
        check=lambda resp, at: contains_any(resp, "Wilfrid Laurier"),
        category="Basic Conversation",
    ),
    TestCase(
        name="Greeting - hello",
        turns=["hello"],
        check=lambda resp, at: contains_any(resp, "Wilfrid Laurier"),
        category="Basic Conversation",
    ),
    TestCase(
        name="Small talk - thank you",
        turns=["thank you"],
        check=lambda resp, at: is_non_empty_response(resp) and not is_off_topic_decline(resp),
        category="Basic Conversation",
    ),
    TestCase(
        name="Small talk - how are you",
        turns=["how are you"],
        check=lambda resp, at: is_non_empty_response(resp) and not is_off_topic_decline(resp),
        category="Basic Conversation",
    ),
    TestCase(
        name="Small talk - who are you",
        turns=["who are you"],
        check=lambda resp, at: contains_any(resp, "Laurier", "WLU", "assistant"),
        category="Basic Conversation",
    ),
]

CONVERSATION_MEMORY_TESTS = [
    TestCase(
        name="Follow-up after course lookup",
        turns=["CP640", "tell me more"],
        check=lambda resp, at: contains_any(resp, "CP640", "Machine Learning"),
        category="Conversation Memory",
    ),
    TestCase(
        name="Follow-up after department lookup",
        turns=["Tell me about the Economics department.", "tell me more"],
        check=lambda resp, at: contains_any(resp, "Economics"),
        category="Conversation Memory",
    ),
    TestCase(
        name="Follow-up after program lookup",
        turns=["What are the requirements for the Master of Business Administration?", "explain"],
        check=lambda resp, at: contains_any(resp, "Business Administration", "MBA"),
        category="Conversation Memory",
    ),
    TestCase(
        name="Follow-up with no prior context does not crash",
        turns=["tell me more"],
        check=lambda resp, at: is_non_empty_response(resp),
        category="Conversation Memory",
    ),
]

PROGRAM_RETRIEVAL_TESTS = [
    TestCase(
        name="MBA admission requirements",
        turns=["What are the admission requirements for the Master of Business Administration?"],
        check=lambda resp, at: contains_any(resp, "admission", "requirement"),
        category="Program Retrieval",
    ),
    TestCase(
        name="Master of Social Work program requirements",
        turns=["Tell me about the Master of Social Work program requirements."],
        check=lambda resp, at: contains_any(resp, "Social Work"),
        category="Program Retrieval",
    ),
    TestCase(
        name="Master of Applied Computing program",
        turns=["What is the Master of Applied Computing program?"],
        check=lambda resp, at: contains_any(resp, "Applied Computing", "Computing"),
        category="Program Retrieval",
    ),
]

PROGRAM_ALIAS_TESTS = [
    TestCase(
        name="MBA alias",
        turns=["What is the MBA program at Laurier?"],
        check=lambda resp, at: contains_any(resp, "MBA", "Business Administration"),
        category="Program Aliases",
    ),
    TestCase(
        name="MSW alias",
        turns=["Tell me about the MSW program."],
        check=lambda resp, at: contains_any(resp, "MSW", "Social Work"),
        category="Program Aliases",
    ),
    TestCase(
        name="MEd alias",
        turns=["What is the MEd program?"],
        check=lambda resp, at: contains_any(resp, "MEd", "Education"),
        category="Program Aliases",
    ),
]

PROGRAM_COMPARISON_TESTS = [
    TestCase(
        name="Compare MBA and Master of Finance",
        turns=[
            "What is the difference between the Master of Business "
            "Administration and the Master of Finance programs?"
        ],
        check=lambda resp, at: (
            contains_any(resp, "Business Administration", "MBA")
            and contains_any(resp, "Finance")
        ),
        category="Program Comparison",
    ),
    TestCase(
        name="Compare Master of Computer Science and Master of Applied Computing",
        turns=[
            "Compare the Master of Computer Science and Master of "
            "Applied Computing programs."
        ],
        check=lambda resp, at: (
            contains_any(resp, "Computer Science")
            and contains_any(resp, "Applied Computing")
        ),
        category="Program Comparison",
    ),
]

FACULTY_RETRIEVAL_TESTS = [
    TestCase(
        name="Economics department",
        turns=["Tell me about the Economics department."],
        check=lambda resp, at: contains_any(resp, "Economics"),
        category="Faculty Retrieval",
    ),
    TestCase(
        name="Biology department's faculty",
        turns=["What faculty is the Biology department part of?"],
        check=lambda resp, at: contains_any(resp, "Science"),
        category="Faculty Retrieval",
    ),
    TestCase(
        name="Sociology department",
        turns=["Tell me about the Sociology department."],
        check=lambda resp, at: contains_any(resp, "Sociology"),
        category="Faculty Retrieval",
    ),
]

RESEARCH_TOPIC_TESTS = [
    TestCase(
        name="Who researches machine learning",
        turns=["Who researches machine learning?"],
        check=lambda resp, at: (
            contains_any(resp, "Azam Asilian Bidgoli", "Yang Liu", "Lei Gao", "Emad Mohammed")
            and not is_off_topic_decline(resp)
        ),
        category="Research Topic",
    ),
    TestCase(
        name="Who researches artificial intelligence",
        turns=["Who researches artificial intelligence?"],
        check=lambda resp, at: (
            contains_any(resp, "Lei Gao", "Samuel Okegbile", "Sukhjit Singh Sehra", "Emad Mohammed")
            and not is_off_topic_decline(resp)
        ),
        category="Research Topic",
    ),
    TestCase(
        name="Who researches consumer behavior",
        turns=["Who researches consumer behavior?"],
        check=lambda resp, at: (
            contains_any(resp, "Hae Joo Kim", "Sarah J. S. Wilner", "Sarah Wilner")
            and not is_off_topic_decline(resp)
        ),
        category="Research Topic",
    ),
    TestCase(
        name="Who researches quantum computing",
        turns=["Who researches quantum computing?"],
        check=lambda resp, at: (
            contains_any(resp, "Alexei Kaltchenko", "Li Wei", "Shohini Ghose")
            and not is_off_topic_decline(resp)
        ),
        category="Research Topic",
    ),
]

FACULTY_COURSES_TAUGHT_TESTS = [
    TestCase(
        name="Who has taught a course code (CP104)",
        turns=["Who has taught CP104?"],
        check=lambda resp, at: (
            contains_any(resp, "CP104")
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Who has taught a course name (Operating Systems)",
        turns=["Who has taught Operating Systems?"],
        check=lambda resp, at: (
            contains_any(resp, "Operating Systems", "CP386")
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Bare course-code lookup still works unchanged",
        turns=["What is CP104?"],
        check=lambda resp, at: (
            contains_any(resp, "CP104")
            and "has been taught by" not in resp.lower()
            and "have taught" not in resp.lower()
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Graceful no-record response for an unknown course code",
        turns=["Who has taught CP999?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Graceful no-match response for an unknown course name",
        turns=["Who has taught Advanced Underwater Basket Weaving?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
]

# Real, verified data (see comments) - not hypothetical. Kaiyu Li's
# actual "Courses Taught" data includes CP600 (Practical Algorithm
# Design) and CP612 (Data Management and Analysis); Emad Mohammed's
# includes CP468 (Artificial Intelligence).
PERSON_TOPIC_COURSES_TAUGHT_TESTS = [
    TestCase(
        name="Literal topic match (algorithm -> CP600)",
        turns=["Has Kaiyu Li taught algorithm courses?"],
        check=lambda resp, at: (
            contains_any(resp, "CP600", "Algorithm")
            and not is_off_topic_decline(resp)
        ),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="Synonym match (AI -> Artificial Intelligence via alias)",
        turns=["Has Emad Mohammed taught any AI courses?"],
        check=lambda resp, at: (
            contains_any(resp, "CP468", "Artificial Intelligence")
            and not is_off_topic_decline(resp)
        ),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="No topical match for a resolved person with real course history",
        turns=["Has Kaiyu Li taught networking courses?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="No course-history available for a resolved person",
        turns=["Has Shohini Ghose taught any AI courses?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="Unknown/unresolvable faculty member",
        turns=["Has John Q Nonexistentperson taught database courses?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Person + Topic Courses Taught",
    ),
]

DEPARTMENT_FALSE_POSITIVE_TESTS = [
    TestCase(
        name="Coffee history is not the History department",
        turns=["Tell me about the history of coffee."],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="Speaking English is not the English department",
        turns=["Do you speak English?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="General philosophy is not the Philosophy department",
        turns=["What is the philosophy behind this decision?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="Loving music is not the Music department",
        turns=["I love music."],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="Common-sense psychology is not the Psychology department",
        turns=["This is just common sense psychology."],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="'History department' still resolves correctly",
        turns=["Tell me about the History department"],
        check=lambda resp, at: (
            contains_any(resp, "History")
            and not is_off_topic_decline(resp)
        ),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="'X at Laurier' phrasing still resolves correctly",
        turns=["History at Laurier"],
        check=lambda resp, at: (
            contains_any(resp, "History")
            and not is_off_topic_decline(resp)
        ),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="'X program' phrasing still resolves correctly",
        turns=["English program"],
        check=lambda resp, at: (
            contains_any(resp, "English")
            and not is_off_topic_decline(resp)
        ),
        category="Department False-Positive Prevention",
    ),
    TestCase(
        name="'Faculty of X' phrasing still resolves correctly",
        turns=["Faculty of Music"],
        check=lambda resp, at: (
            contains_any(resp, "Music")
            and not is_off_topic_decline(resp)
        ),
        category="Department False-Positive Prevention",
    ),
]

OUT_OF_DOMAIN_TESTS = [
    TestCase(
        name="Sports question",
        turns=["Tell me about the latest Super Bowl champion."],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
    TestCase(
        name="Celebrity / movie question",
        turns=["What's your favorite movie?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
    TestCase(
        name="Coding question",
        turns=["Can you write a Python function to sort a list?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
    TestCase(
        name="Politics question",
        turns=["Who is the president of the United States?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
    TestCase(
        name="General knowledge question",
        turns=["What's the weather like today?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
    TestCase(
        name="Control - in-domain tuition question should NOT be blocked",
        turns=["What is the tuition for graduate programs at WLU?"],
        check=lambda resp, at: not is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
    ),
]

CATEGORIES = [
    ("Basic Conversation", "Basic Conversation", BASIC_CONVERSATION_TESTS),
    ("Conversation Memory", "Conversation Memory", CONVERSATION_MEMORY_TESTS),
    ("Program Retrieval", "Program Retrieval", PROGRAM_RETRIEVAL_TESTS),
    ("Program Aliases", "Aliases", PROGRAM_ALIAS_TESTS),
    ("Program Comparison", "Comparison", PROGRAM_COMPARISON_TESTS),
    ("Faculty Retrieval", "Faculty", FACULTY_RETRIEVAL_TESTS),
    ("Research Topic", "Research Topic", RESEARCH_TOPIC_TESTS),
    ("Faculty Courses Taught", "Courses Taught", FACULTY_COURSES_TAUGHT_TESTS),
    ("Person + Topic Courses Taught", "Person+Topic", PERSON_TOPIC_COURSES_TAUGHT_TESTS),
    ("Department False-Positive Prevention", "Dept False-Positive", DEPARTMENT_FALSE_POSITIVE_TESTS),
    ("Out-of-Domain Detection", "Out-of-Domain", OUT_OF_DOMAIN_TESTS),
]


# -----------------------------
# Runner
# -----------------------------

def run_test(test: TestCase) -> bool:
    print(f"Test: {test.name}")
    print(f"Category: {test.category}")

    if len(test.turns) > 1:
        print(f"Setup: {' -> '.join(test.turns[:-1])}")

    print(f"Question: {test.turns[-1]}")

    try:
        response = run_turns(test.turns)
        passed = test.check(response, None)
    except Exception as e:
        response = f"ERROR: {e}"
        passed = False

    print(f"Response: {response}")
    print("PASS" if passed else "FAIL")
    print("-" * 60)

    return passed


# -----------------------------
# Data integrity checks (Sprint 6B)
#
# These guard against the faculty-profile URL-variance bug (Sprint 6A
# audit issue #1) reappearing: www vs non-www, trailing slashes, stray
# query parameters, legacy URL templates, and the malformed
# "faculties..." pattern all used to defeat get_faculty_links.py's
# exact-URL dedup and produce duplicate faculty rows for the same
# person. Not AppTest-based like the categories above - these check the
# scraper's URL normalization directly and the resulting database state,
# not chatbot conversation behavior.
# -----------------------------

def _check_url_normalization_collapses_variants():

    import sqlite3

    sys.path.insert(0, str(SRC_DIR))
    from get_faculty_links import _normalize_profile_url

    # One representative pair per variant type confirmed in the Sprint
    # 6A audit - each pair must normalize to the identical canonical URL.
    variant_pairs = [
        (
            "www vs non-www",
            "https://wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/noam-miller/index.html",
            "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/noam-miller/index.html",
        ),
        (
            "stray query parameter",
            "https://www.wlu.ca/academics/faculties/faculty-of-liberal-arts/faculty-profiles/kelly-gallagher-mackay/index.html?ref=faculty-profiles%2Fliberal-arts%2Fkelly-gallagher-mackay.html",
            "https://www.wlu.ca/academics/faculties/faculty-of-liberal-arts/faculty-profiles/kelly-gallagher-mackay/index.html",
        ),
        (
            "legacy URL template",
            "https://www.wlu.ca/faculty-profiles/science/mihai-costea.html",
            "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/mihai-costea/index.html",
        ),
        (
            "malformed 'faculties...' pattern",
            "https://www.wlu.ca/academics/faculties.../faculty-of-science/faculty-profiles/diane-gregory/index.html",
            "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/diane-gregory/index.html",
        ),
        (
            "stale current-pattern path (moved faculty)",
            "https://wlu.ca/academics/faculties/faculty-of-arts/faculty-profiles/philip-marsh/index.html",
            "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/philip-marsh/index.html",
        ),
    ]

    for label, url_a, url_b in variant_pairs:

        try:
            norm_a = _normalize_profile_url(url_a)
            norm_b = _normalize_profile_url(url_b)
        except Exception as e:
            yield (f"URL normalization: {label}", False, f"ERROR: {e}")
            continue

        yield (
            f"URL normalization: {label}",
            norm_a == norm_b,
            f"{norm_a!r} vs {norm_b!r}"
        )

    # Trailing slash, checked without any live request involved.
    no_slash = _normalize_profile_url(
        "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/noam-miller/index.html"
    )
    with_slash = _normalize_profile_url(
        "https://www.wlu.ca/academics/faculties/faculty-of-science/faculty-profiles/noam-miller/index.html/"
    )
    yield (
        "URL normalization: trailing slash",
        no_slash == with_slash,
        f"{no_slash!r} vs {with_slash!r}"
    )


def _check_no_duplicate_faculty_records():

    import sqlite3

    conn = sqlite3.connect(str(ROOT_DIR / "data" / "faculty.db"))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, COUNT(*) c FROM faculty GROUP BY name HAVING c > 1"
    )
    duplicate_names = cursor.fetchall()

    cursor.execute(
        "SELECT source_url, COUNT(*) c FROM faculty GROUP BY source_url HAVING c > 1"
    )
    duplicate_urls = cursor.fetchall()

    conn.close()

    yield (
        "No duplicate faculty names in faculty.db",
        len(duplicate_names) == 0,
        f"duplicates found: {duplicate_names}" if duplicate_names else ""
    )

    yield (
        "No duplicate faculty source_urls in faculty.db",
        len(duplicate_urls) == 0,
        f"duplicates found: {duplicate_urls}" if duplicate_urls else ""
    )


def run_data_integrity_checks():

    checks = list(_check_url_normalization_collapses_variants())
    checks += list(_check_no_duplicate_faculty_records())

    passed_count = 0

    for name, passed, detail in checks:
        print(f"Check: {name}")
        if detail:
            print(f"Detail: {detail}")
        print("PASS" if passed else "FAIL")
        print("-" * 60)
        if passed:
            passed_count += 1

    return passed_count, len(checks)


def main():
    if not (ROOT_DIR / "data" / "courses.db").exists():
        print(
            "ERROR: data/courses.db not found. "
            "Run this script from the project root, e.g.:\n"
            "  python3 src/evaluate.py"
        )
        sys.exit(1)

    scores = {}

    for category, _, tests in CATEGORIES:
        passed_count = 0

        for test in tests:
            if run_test(test):
                passed_count += 1

        scores[category] = (passed_count, len(tests))

    integrity_passed, integrity_total = run_data_integrity_checks()

    print("\n" + "=" * 36)
    print("Evaluation Summary")
    print("=" * 36 + "\n")

    total_passed = 0
    total_count = 0

    for category, label, _ in CATEGORIES:
        passed_count, count = scores[category]
        total_passed += passed_count
        total_count += count
        print(f"{label}: {passed_count}/{count}")

    total_passed += integrity_passed
    total_count += integrity_total
    print(f"Data Integrity: {integrity_passed}/{integrity_total}")

    print(f"\nOverall Score: {total_passed}/{total_count}")


if __name__ == "__main__":
    main()
