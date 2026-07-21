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


def _check_email_extraction():

    from bs4 import BeautifulSoup
    from load_faculty import _extract_email

    def extract(html):
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        return _extract_email(soup, text)

    cases = [
        (
            "Email extraction: mailto href with no visible address text",
            '<a href="mailto:jbenhamrennick@wlu.ca">Email</a>',
            "jbenhamrennick@wlu.ca",
        ),
        (
            "Email extraction: [at]/[dot] obfuscated visible text",
            '<p>Contact: jaguinaldo [at] wlu.ca</p>',
            "jaguinaldo@wlu.ca",
        ),
        (
            "Email extraction: [at]/[dot] obfuscated inside mailto href itself",
            '<a href="mailto:pmallet [at] wlu [dot] ca">Email</a>',
            "pmallet@wlu.ca",
        ),
        (
            "Email extraction: *.wlu.ca subdomain (not just literal wlu.ca)",
            '<a href="mailto:Eli.Teram@ret.wlu.ca">Email</a>',
            "eli.teram@ret.wlu.ca",
        ),
        (
            "Email extraction: non-wlu.ca mailto is not surfaced",
            '<a href="mailto:someone@gmail.com">Email</a>',
            "",
        ),
        (
            "Email extraction: no email present yields empty string",
            '<p>No contact information here.</p>',
            "",
        ),
    ]

    for name, html, expected in cases:
        result = extract(html)
        yield (
            name,
            result.lower() == expected.lower(),
            f"got {result!r}, expected {expected!r}"
        )


def _check_image_link_rejection():

    from get_faculty_links import _resolve_profile_href

    cases = [
        (
            "Image-link rejection: .jpg resolves to index.html in same directory",
            "faculty-profiles/abderrahman-beggar/faculty_arts_abderrahman_beggar.jpg",
            "faculty-profiles/abderrahman-beggar/index.html",
        ),
        (
            "Image-link rejection: other image extensions (.png, .jpeg, query string)",
            "faculty-profiles/some-name/pic.jpeg?v=2",
            "faculty-profiles/some-name/index.html",
        ),
        (
            "Valid profile acceptance: a normal index.html href is unaffected",
            "faculty-profiles/some-name/index.html",
            "faculty-profiles/some-name/index.html",
        ),
    ]

    for name, href, expected in cases:
        result = _resolve_profile_href(href)
        yield (name, result == expected, f"got {result!r}, expected {expected!r}")


def _check_beggar_recovered():

    import sqlite3

    conn = sqlite3.connect(str(ROOT_DIR / "data" / "faculty.db"))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT title, email, source_url FROM faculty "
        "WHERE name = 'Abderrahman Beggar'"
    )
    row = cursor.fetchone()

    conn.close()

    if not row:
        yield ("Abderrahman Beggar resolves to a real profile", False, "no row found")
        return

    title, email, source_url = row

    yield (
        "Abderrahman Beggar resolves to a real profile (not an image URL)",
        bool(title) and not source_url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")
        ),
        f"title={title!r}, email={email!r}, source_url={source_url!r}"
    )


def _check_course_prerequisites():

    import sqlite3

    conn = sqlite3.connect(str(ROOT_DIR / "data" / "courses.db"))
    cursor = conn.cursor()

    # Prerequisite extraction: a known multi-code prerequisite (CP312)
    # extracted correctly into both the raw text and the derived
    # reference table.
    cursor.execute(
        "SELECT prerequisites_text FROM courses WHERE course_code = 'CP312'"
    )
    row = cursor.fetchone()
    cp312_text = row[0] if row else ""

    yield (
        "Prerequisite extraction: CP312 raw text mentions its known prerequisites",
        all(code in cp312_text for code in ["CP264", "CP114", "CP213", "CP214", "MA238"]),
        f"got {cp312_text!r}"
    )

    # Hyperlink formatting: no fragmented text from newline-joining
    # around the inline course-code links, and no stray space-before-
    # punctuation artifacts from space-joining.
    yield (
        "Hyperlink formatting: prerequisite text has no embedded newlines",
        "\n" not in cp312_text,
        f"got {cp312_text!r}"
    )

    yield (
        "Hyperlink formatting: no stray space before closing punctuation",
        " )" not in cp312_text and " ." not in cp312_text and " ;" not in cp312_text,
        f"got {cp312_text!r}"
    )

    cursor.execute(
        "SELECT course_code FROM course_prerequisite_refs "
        "WHERE course_code = 'CP312' ORDER BY required_course_code"
    )
    cp312_refs = [r[0] for r in cursor.fetchall()]

    yield (
        "Prerequisite extraction: CP312 has derived reference rows",
        len(cp312_refs) >= 5,
        f"found {len(cp312_refs)} reference rows for CP312"
    )

    # Fused-label handling: a course whose page fuses "Co-requisites" and
    # "Prerequisites" into one unseparated label populates both fields
    # with the same text, rather than being missed entirely by an exact
    # label match.
    cursor.execute(
        "SELECT prerequisites_text, corequisites_text FROM courses "
        "WHERE source_url LIKE '%c=75171%'"
    )
    row = cursor.fetchone()

    yield (
        "Fused-label handling: RE407's fused Co-requisites/Prerequisites label populates both fields",
        bool(row) and bool(row[0]) and row[0] == row[1],
        f"got {row!r}"
    )

    conn.close()

    # Direct lookup, reverse lookup, and no-prerequisite fallback -
    # exercised through structured_search(), not a raw DB query, since
    # these are retrieval-layer behaviors (intent detection + graceful
    # fallback wording), not just data presence.
    sys.path.insert(0, str(SRC_DIR))
    from retriever import structured_search

    result = structured_search("What are the prerequisites for CP600?", {})
    yield (
        "Direct prerequisite lookup: CP600 (no prerequisites listed)",
        bool(result) and "no prerequisites" in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )

    result = structured_search("What are the prerequisites for CP312?", {})
    yield (
        "Direct prerequisite lookup: CP312 (has prerequisites listed)",
        bool(result) and "CP264" in result[0],
        f"got {result[0] if result else None!r}"
    )

    result = structured_search("Which courses require CP264?", {})
    yield (
        "Reverse prerequisite lookup: courses requiring CP264",
        bool(result) and "CP312" in result[0],
        f"got {result[0] if result else None!r}"
    )

    result = structured_search("Does CP312 require CP264?", {})
    yield (
        "Relational prerequisite lookup: does CP312 require CP264",
        bool(result) and "yes" in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )

    result = structured_search("What courses have no prerequisites listed?", {})
    yield (
        "No-prerequisite fallback: listing courses with none listed",
        bool(result) and "no prerequisite" in result[0].lower(),
        f"got {result[0][:100] if result else None!r}"
    )


def _check_contextual_reference_resolution():

    sys.path.insert(0, str(SRC_DIR))
    from retriever import resolve_contextual_reference, structured_search

    empty_memory = {
        "last_course": None, "last_program": None,
        "last_department": None, "last_faculty": None,
    }

    # A completely empty memory is treated as "not applicable" (returns
    # None, not a clarification) - see the guard at the top of
    # resolve_contextual_reference(): with truly nothing established,
    # a bare reference word is far more likely to be ordinary grammar in
    # an unrelated sentence than a genuine follow-up. So "unresolved"
    # here means a real multi-turn scenario: something IS in memory
    # (matching how these cases actually arise in practice - e.g. a
    # course was already discussed), but not the specific type this
    # question needs, and the placeholder value is deliberately
    # unmatchable so resolution genuinely fails rather than coincidentally
    # succeeding.
    partially_populated_memory = dict(empty_memory)
    partially_populated_memory["last_department"] = "Placeholder Department 999"

    unresolved_cases = [
        ("unresolved 'it'", "Does it have prerequisites?"),
        ("unresolved 'they'", "What research do they do?"),
        ("unresolved 'that professor'", "Does that professor do AI research?"),
        ("unresolved 'that course'", "What are its requirements?"),
        ("unresolved comparison", "Compare those."),
        ("unresolved ordinal reference", "Tell me about the second one."),
    ]

    for label, question in unresolved_cases:
        result = resolve_contextual_reference(question, dict(partially_populated_memory))
        yield (
            f"Contextual reference: {label} produces a clarification",
            bool(result) and result[0] == "clarify",
            f"got {result!r}"
        )

    # Resolved cases: with real memory context, the same style of
    # question is answered correctly rather than clarified or
    # hallucinated - verified against known real data (CP312's actual
    # prerequisites, confirmed elsewhere in this suite).
    course_memory = dict(empty_memory)
    course_memory["last_course"] = "CP312"

    result = resolve_contextual_reference("Does it have prerequisites?", course_memory)
    yield (
        "Contextual reference: 'it' resolves to CP312's real prerequisites",
        bool(result) and result[0] == "resolved" and "CP264" in result[1],
        f"got {result[0] if result else None!r}"
    )

    result = resolve_contextual_reference("Who teaches it?", dict(course_memory))
    yield (
        "Contextual reference: 'it' resolves to CP312's real courses-taught data",
        bool(result) and result[0] == "resolved",
        f"got {result[:1] if result else None!r}"
    )

    # Standalone questions must never be intercepted - structured_search
    # already succeeds on these, so resolve_contextual_reference is never
    # even reached in the real app.py routing (it's only called after
    # structured_search returns None). Verified directly here.
    standalone_cases = [
        "What is CP600?",
        "Who is Tripat Gill?",
        "Who works in Marketing?",
    ]

    for question in standalone_cases:
        already_resolved = structured_search(question, dict(empty_memory))
        yield (
            f"Standalone question bypasses clarification: {question!r}",
            bool(already_resolved),
            "structured_search already resolves this without needing "
            "the contextual-reference gate at all"
        )


def _check_program_course_requirements():

    import sqlite3

    sys.path.insert(0, str(SRC_DIR))
    from retriever import structured_search

    # Required-course extraction: a known graduate program (verified live
    # during Sprint 7D implementation) has its real required courses
    # captured, keyed off the embedded course hyperlinks.
    conn = sqlite3.connect(str(ROOT_DIR / "data" / "programs.db"))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT course_code FROM program_course_requirements "
        "WHERE program_name = 'Master of Applied Computing'"
    )
    mac_codes = {row[0] for row in cursor.fetchall()}
    conn.close()

    yield (
        "Required-course extraction: Master of Applied Computing includes CP600",
        "CP600" in mac_codes,
        f"got {mac_codes!r}"
    )

    # Reverse lookup: which graduate programs require CP600 - known to
    # include both MAC and Master of Computer Science.
    result = structured_search("Which graduate programs require CP600?", {})
    yield (
        "Reverse lookup: programs requiring CP600 include MAC and MCS",
        bool(result)
        and "Master of Applied Computing" in result[0]
        and "Master of Computer Science" in result[0],
        f"got {result[0] if result else None!r}"
    )

    # Program lookup: does a known program require a known course, and
    # does it correctly say no for a course it doesn't require.
    result = structured_search(
        "Does the Master of Applied Computing require CP600?", {}
    )
    yield (
        "Program lookup: MAC does require CP600",
        bool(result) and "yes" in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )

    result = structured_search(
        "Does the Master of Applied Computing require CP601?", {}
    )
    yield (
        "Program lookup: MAC does not require CP601",
        bool(result) and "not listed" in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )

    result = structured_search(
        "Which required courses are listed for the Master of Applied Computing?",
        {}
    )
    yield (
        "Program lookup: required courses listed for MAC",
        bool(result) and "CP600" in result[0],
        f"got {result[0] if result else None!r}"
    )

    # Graceful fallback: a course with no graduate program requiring it.
    result = structured_search("Which graduate programs require CP999?", {})
    yield (
        "Graceful fallback: no program requires a nonexistent course code",
        bool(result) and "no graduate program" in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )

    # Undergraduate exclusion: an undergraduate program name should never
    # resolve through this feature - it must fall through to whatever
    # existing behavior handles the query instead (never a fabricated
    # undergraduate mapping claim).
    conn = sqlite3.connect(str(ROOT_DIR / "data" / "programs.db"))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT program_name FROM program_course_requirements"
    )
    requirement_program_names = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        "SELECT program_name FROM programs WHERE level = 'undergraduate'"
    )
    undergrad_names = {row[0] for row in cursor.fetchall()}
    conn.close()

    yield (
        "Undergraduate exclusion: no undergraduate program name in program_course_requirements",
        len(undergrad_names & requirement_program_names) == 0,
        f"overlap: {undergrad_names & requirement_program_names!r}"
    )

    result = structured_search(
        "Does the Honours Bachelor of Business Administration require CP104?",
        {}
    )
    yield (
        "Undergraduate exclusion: no fabricated undergraduate program-requirement claim",
        not bool(result) or "required course" not in result[0].lower(),
        f"got {result[0] if result else None!r}"
    )


def run_data_integrity_checks():

    checks = list(_check_url_normalization_collapses_variants())
    checks += list(_check_no_duplicate_faculty_records())
    checks += list(_check_email_extraction())
    checks += list(_check_course_prerequisites())
    checks += list(_check_image_link_rejection())
    checks += list(_check_beggar_recovered())
    checks += list(_check_contextual_reference_resolution())
    checks += list(_check_program_course_requirements())

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
