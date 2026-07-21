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

    print(f"\nOverall Score: {total_passed}/{total_count}")


if __name__ == "__main__":
    main()
