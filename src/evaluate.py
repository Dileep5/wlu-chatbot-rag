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
    # Which of the four Sprint 8B report buckets this test's pass/fail
    # rolls up into. Defaults to "retrieval" so every pre-existing
    # TestCase (added before this field existed) needs no changes -
    # nearly all of them genuinely are retrieval-accuracy checks.
    # "clarification" = expected outcome is a clarification response;
    # "unsupported" = expected outcome is a graceful decline/non-answer
    # for an out-of-scope question, with hallucination explicitly ruled
    # out.
    metric: str = "retrieval"


# -----------------------------
# Response helpers
#
# Sprint 10A audit (see investigation report for full detail): every
# helper below was reviewed for brittle substring matching that could
# produce FALSE POSITIVES - a check failing a genuinely correct answer
# because of coincidental wording, not because anything is actually
# wrong. Only one was found to have a confirmed, reproduced case of this
# (is_off_topic_decline, fixed below). The others are documented as
# audited-and-sound rather than left unexamined:
#
# - contains_any()/contains_all(): substring matching, but every keyword
#   used against them project-wide is a specific proper noun, course
#   code, or distinctive multi-word phrase (checked directly against
#   every call site in this file) - there is no coincidental-collision
#   risk here comparable to is_off_topic_decline's two-generic-words
#   problem, so no change was made.
# - is_non_empty_response(): a permissive lower bound (non-empty, not a
#   literal error string) - it can never wrongly FAIL a correct answer,
#   only under-verify an incorrect one, which is a different (and
#   already separately covered by contains_any's factual-content checks)
#   concern than what this audit targets.
# - is_clarification_response(): checks for "i'm not sure", but every
#   clarification message retriever.py produces is a fixed, hardcoded
#   string returned with NO LLM involvement (never passed through
#   generate_answer()) and is a deliberate, established sentinel phrase
#   (Sprint 7B/9B), not a coincidental heuristic guess - unlike
#   is_off_topic_decline, the true side here is 100% deterministic.
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
    # Previously: "wilfrid laurier" in lower and "only" in lower. This
    # heuristic produced confirmed, reproduced false positives (Sprint
    # 9B/9C investigations) whenever a genuinely correct, grounded answer
    # happened to contain the word "only" for an unrelated reason - e.g.
    # scraped department text ("Only students who are graduates...") or
    # a retrieval function's own scope disclaimer ("This only covers
    # explicitly required courses..."). Both confirmed cases involved
    # answers that also naturally mention "Wilfrid Laurier University",
    # so the two-generic-word co-occurrence check was never a reliable
    # signal of an off-topic decline specifically.
    #
    # The off-topic decline is actually a FIXED, hardcoded constant
    # (app.py's OFF_TOPIC_MESSAGE) returned verbatim with no LLM
    # involvement whatsoever - the off-topic branch in app.py assigns it
    # directly (`answer = OFF_TOPIC_MESSAGE`), never through
    # generate_answer(). Checking for the actual constant text is
    # therefore not a heuristic at all: it deterministically matches
    # exactly and only when that exact branch fired, eliminating the
    # false-positive class entirely. Imported lazily (matching this
    # file's existing pattern for retriever/get_faculty_links/
    # load_faculty) rather than at module level, so importing app.py's
    # side effects only ever happen after an AppTest has already run one.
    from app import OFF_TOPIC_MESSAGE

    return OFF_TOPIC_MESSAGE in text


def is_clarification_response(text: str) -> bool:
    # All clarification messages resolve_contextual_reference() produces
    # (course/program/department/faculty/generic/compare) begin "I'm not
    # sure" - a deliberate, established sentinel phrase (Sprint 7B/9B),
    # not a coincidental heuristic guess, and every one of them bypasses
    # the LLM entirely (assigned directly, never through
    # generate_answer()) - this is deliberately the same literal-text
    # check used throughout Sprint 7B's own verification, kept here as a
    # single shared helper rather than re-deriving it per test.
    return "i'm not sure" in text.lower()


def is_graceful_fallback(text: str) -> bool:
    """Capability-aware validator for 'this should decline/fall back
    gracefully rather than fabricate' tests (Sprint 10A) - replaces the
    repeated inline `is_non_empty_response(resp) and not
    is_off_topic_decline(resp)` lambda pattern used across every
    'unsupported' metric test with one named helper expressing what's
    actually being verified: a real, non-empty response was produced,
    and it wasn't misrouted into an off-topic decline (using the fixed,
    non-heuristic check above)."""

    return is_non_empty_response(text) and not is_off_topic_decline(text)


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
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Faculty Courses Taught",
        metric="unsupported",
    ),
    TestCase(
        name="Graceful no-match response for an unknown course name",
        turns=["Who has taught Advanced Underwater Basket Weaving?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Faculty Courses Taught",
        metric="unsupported",
    ),

    # Present-tense instructor intent (Sprint 10C): "who teaches X" now
    # resolves through the same instructor lookup as "who has taught X",
    # rather than silently falling through to the plain course-lookup
    # branch and returning the generic course description instead of
    # answering who teaches it - confirmed via direct structured_search()
    # testing to be the actual prior behavior before this fix.
    TestCase(
        name="Present tense - who teaches a course code (CP312)",
        turns=["Who teaches CP312?"],
        check=lambda resp, at: (
            contains_any(resp, "Foley", "Ebrahimi")
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Present tense - who teaches a course code (BU111)",
        turns=["Who teaches BU111?"],
        check=lambda resp, at: (
            contains_any(resp, "Brandon Van Dam", "Van Dam")
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Present tense - who teaches a course name (Operating Systems)",
        turns=["Who teaches Operating Systems?"],
        check=lambda resp, at: (
            contains_any(resp, "Operating Systems", "CP386")
            and not is_off_topic_decline(resp)
        ),
        category="Faculty Courses Taught",
    ),
    TestCase(
        name="Present tense - graceful no-record response for an unknown course code",
        turns=["Who teaches CP999?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Faculty Courses Taught",
        metric="unsupported",
    ),
    TestCase(
        # Negative case: "who teaches in/at X" is department/faculty-list
        # phrasing (a different capability), not a course-instructor
        # question - must NOT be hijacked by the broadened present-tense
        # pattern. This is the exact collision the original code
        # comment warned about, now guarded by the negative lookahead.
        name="Negative case: 'who teaches in Marketing' still resolves to the department list, not course lookup",
        turns=["Who teaches in Marketing?"],
        check=lambda resp, at: (
            contains_any(resp, "Marketing")
            and "no course matching" not in resp.lower()
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
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="No course-history available for a resolved person",
        turns=["Has Shohini Ghose taught any AI courses?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Person + Topic Courses Taught",
    ),
    TestCase(
        name="Unknown/unresolvable faculty member",
        turns=["Has John Q Nonexistentperson taught database courses?"],
        check=lambda resp, at: is_graceful_fallback(resp),
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

# Protects: the program->department coordinator join (Sprint 4B), via
# the full chatbot rather than a direct structured_search() call - the
# exact end-to-end coverage gap Sprint 8A found missing. Pass criteria:
# a program with real coordinator data names the actual coordinator(s);
# a program whose joined department has no coordinator on file declines
# gracefully rather than guessing a name.
COORDINATOR_LOOKUP_TESTS = [
    TestCase(
        name="MAC coordinator - real data present",
        turns=["Who is the program coordinator for the Master of Applied Computing?"],
        check=lambda resp, at: (
            contains_any(resp, "Dariush Ebrahimi", "Usama Mir")
            and not is_off_topic_decline(resp)
        ),
        category="Coordinator Lookup",
    ),
    TestCase(
        name="MCS coordinator - same department join as MAC",
        turns=["Who is the program coordinator for the Master of Computer Science?"],
        check=lambda resp, at: (
            contains_any(resp, "Dariush Ebrahimi", "Usama Mir")
            and not is_off_topic_decline(resp)
        ),
        category="Coordinator Lookup",
    ),
    TestCase(
        name="MBA coordinator - graceful fallback (no data on file)",
        turns=["Who coordinates the MBA?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Coordinator Lookup",
        metric="unsupported",
    ),
    TestCase(
        name="Non-coordinator program question unaffected",
        turns=["What is the Master of Applied Computing program?"],
        check=lambda resp, at: contains_any(resp, "Applied Computing", "Computing"),
        category="Coordinator Lookup",
    ),
]

# Protects: direct department coordinator lookup (Sprint 10E), via the
# full chatbot. Pass criteria: a department with real coordinator data
# names the actual coordinator; a department with none declines
# gracefully; a nonexistent department is never fabricated; a follow-up
# question resolves the department's coordinator through entity-history,
# without needing to repeat the department's name.
DEPARTMENT_COORDINATOR_TESTS = [
    TestCase(
        name="History department coordinator - real data present",
        turns=["Who coordinates the History department?"],
        check=lambda resp, at: (
            contains_any(resp, "Susan Neylan")
            and not is_off_topic_decline(resp)
        ),
        category="Department Coordinator",
    ),
    TestCase(
        name="Single-word department name - who is the coordinator of Biology?",
        turns=["Who is the coordinator of Biology?"],
        check=lambda resp, at: (
            contains_any(resp, "Jonathan Wilson")
            and not is_off_topic_decline(resp)
        ),
        category="Department Coordinator",
    ),
    TestCase(
        name="Department without coordinator data - graceful fallback",
        turns=["Who coordinates the Archaeology and Heritage Studies department?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Department Coordinator",
        metric="unsupported",
    ),
    TestCase(
        name="Unsupported/nonexistent department - no fabrication",
        turns=["Who coordinates the Underwater Basketweaving department?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Department Coordinator",
        metric="unsupported",
    ),
    TestCase(
        name="Multi-turn: department coordinator follow-up via entity-history",
        turns=[
            "Tell me about the History department.",
            "Who coordinates it?",
        ],
        check=lambda resp, at: (
            contains_any(resp, "Susan Neylan")
            and not is_clarification_response(resp)
        ),
        category="Department Coordinator",
    ),
]

# Protects: course prerequisite extraction and retrieval (Sprint 6F), via
# the full chatbot. Pass criteria: a course with real prerequisites
# states its actual required codes; a course with none says so plainly;
# reverse and relational lookups return the real, verified answer; a
# nonexistent course code never produces a fabricated answer.
COURSE_PREREQUISITE_TESTS = [
    TestCase(
        name="Direct lookup - CP312 has real, specific prerequisites",
        turns=["What are the prerequisites for CP312?"],
        check=lambda resp, at: contains_any(resp, "CP264"),
        category="Course Prerequisites",
    ),
    TestCase(
        name="Direct lookup - CP600 has no prerequisites listed",
        turns=["What are the prerequisites for CP600?"],
        check=lambda resp, at: contains_any(resp, "no prerequisite", "No prerequisite"),
        category="Course Prerequisites",
    ),
    TestCase(
        name="Reverse lookup - which courses require CP264",
        turns=["Which courses require CP264?"],
        check=lambda resp, at: contains_any(resp, "CP312"),
        category="Course Prerequisites",
    ),
    TestCase(
        name="Relational lookup - CP312 does require CP264",
        turns=["Does CP312 require CP264?"],
        check=lambda resp, at: contains_any(resp, "Yes", "yes"),
        category="Course Prerequisites",
    ),
    TestCase(
        name="Hallucination guard - nonexistent course code",
        turns=["Does CP312 require CP999?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Course Prerequisites",
        metric="unsupported",
    ),
]

# Protects: exposure of existing courses.db metadata (Sprint 10D), via
# the full chatbot rather than a direct structured_search() call. The
# exact field-presence/ordering guarantees are already checked
# deterministically in Data Integrity (_check_course_metadata_exposure);
# these confirm the enrichment actually reaches a real, LLM-phrased
# answer, and that a plain course question with none of these fields
# still answers normally (no regression to existing behavior).
COURSE_METADATA_TESTS = [
    TestCase(
        name="Exclusions surfaced in a real answer (CQ609)",
        turns=["What is CQ609?"],
        check=lambda resp, at: (
            contains_any(resp, "CQ640D", "exclusion")
            and not is_off_topic_decline(resp)
        ),
        category="Course Metadata",
    ),
    TestCase(
        name="Location surfaced in a real answer (UU400)",
        turns=["What is UU400?"],
        check=lambda resp, at: (
            contains_any(resp, "Brantford")
            and not is_off_topic_decline(resp)
        ),
        category="Course Metadata",
    ),
    TestCase(
        name="Plain course question with no metadata fields still answers normally",
        turns=["What is CS600?"],
        check=lambda resp, at: (
            contains_any(resp, "Communication Studies", "Graduate Seminar")
            and not is_off_topic_decline(resp)
        ),
        category="Course Metadata",
    ),
]

# Protects: graduate program required-course retrieval (Sprint 7D), via
# the full chatbot. Pass criteria: direct, reverse, and relational
# lookups all reflect real data; an undergraduate program never produces
# a fabricated course-requirement mapping (that capability doesn't
# exist, and must never be guessed at).
GRADUATE_PROGRAM_REQUIREMENTS_TESTS = [
    TestCase(
        name="Direct lookup - required courses for MAC",
        turns=["Which required courses are listed for the Master of Applied Computing?"],
        check=lambda resp, at: contains_any(resp, "CP600"),
        category="Graduate Program Requirements",
    ),
    TestCase(
        name="Relational lookup - MAC does require CP600",
        turns=["Does the Master of Applied Computing require CP600?"],
        check=lambda resp, at: contains_any(resp, "Yes", "yes"),
        category="Graduate Program Requirements",
    ),
    TestCase(
        name="Reverse lookup - programs requiring CP600",
        turns=["Which graduate programs require CP600?"],
        check=lambda resp, at: contains_all(resp, "Applied Computing")
        and contains_any(resp, "Computer Science"),
        category="Graduate Program Requirements",
    ),
    TestCase(
        name="Graceful fallback - course not required by MAC",
        turns=["Does the Master of Applied Computing require CP601?"],
        check=lambda resp, at: is_graceful_fallback(resp),
        category="Graduate Program Requirements",
        metric="unsupported",
    ),
    TestCase(
        name="Undergraduate exclusion - no fabricated mapping",
        turns=["Does the Honours Bachelor of Business Administration require CP104?"],
        check=lambda resp, at: (
            is_non_empty_response(resp)
            and "required course" not in resp.lower()
        ),
        category="Graduate Program Requirements",
        metric="unsupported",
    ),
]

# Protects: resolve_contextual_reference() (Sprint 7B) and the broader
# multi-turn memory model, end-to-end through real conversations rather
# than isolated function calls - the exact coverage gap Sprint 8A found
# missing (every prior verification of this feature was an ad hoc script,
# never a permanent fixture). Each conversation's pass criteria is
# checked only on its final turn, by design: five conversations, one
# per required theme (successful follow-up, clarification, context
# switching, unsupported follow-up, ambiguity), each engineered so its
# last turn is the one meaningful assertion point.
MULTI_TURN_CONVERSATION_TESTS = [
    TestCase(
        name="Successful follow-up: pronoun resolves to real prerequisite data",
        turns=[
            "Tell me about CP312.",
            "Does it have prerequisites?",
        ],
        check=lambda resp, at: (
            contains_any(resp, "CP264")
            and not is_clarification_response(resp)
        ),
        category="Multi-Turn Conversations",
    ),
    TestCase(
        # This used to clarify: search_faculty_courses_taught() never
        # wrote the resolved instructor back into memory, so "that
        # professor" had nothing to resolve against (Sprint 7A's
        # documented gap). Sprint 9B's entity-history write-back closes
        # exactly this gap - CP312 is really taught by Angele Foley and
        # Dariush Ebrahimi, and "that professor" now correctly resolves
        # to one of them, grounding the answer in their real profile
        # instead of clarifying. This is an intentional, expected
        # behavior change, not a regression.
        name="Multi-hop: course -> instructor -> grounded profile (entity-history write-back)",
        turns=[
            "Tell me about CP312.",
            "Who teaches it?",
            "Does that professor do AI research?",
        ],
        check=lambda resp, at: (
            contains_any(resp, "Foley", "Ebrahimi")
            and not is_off_topic_decline(resp)
            and not is_clarification_response(resp)
        ),
        category="Multi-Turn Conversations",
    ),
    TestCase(
        name="Context switching: 'that course' tracks the most recent course, not the first",
        turns=[
            "Tell me about CP312.",
            "Tell me about CP600.",
            "What are the prerequisites for that course?",
        ],
        check=lambda resp, at: (
            contains_any(resp, "CP600", "no prerequisite", "No prerequisite")
            and "CP264" not in resp
        ),
        category="Multi-Turn Conversations",
    ),
    TestCase(
        # This used to clarify: departments.db's `coordinator` column
        # (already populated for Psychology - Jeffery Jones, PhD) was
        # never exposed by any retrieval path, so "does it have a
        # coordinator?" had nothing to resolve against. Sprint 10E adds
        # direct department coordinator lookup, which closes exactly
        # this gap - "it" (the Psychology department) now correctly
        # resolves and grounds the answer in the real coordinator's
        # name instead of clarifying. An intentional, expected behavior
        # change, not a regression (same precedent as Sprint 9B's
        # courses-taught write-back test update).
        name="Follow-up: department coordinator now resolves via entity-history (Sprint 10E)",
        turns=[
            "Tell me about the Psychology department.",
            "Does it have a coordinator?",
        ],
        check=lambda resp, at: (
            contains_any(resp, "Jeffery Jones", "Jones")
            and not is_clarification_response(resp)
            and not is_off_topic_decline(resp)
        ),
        category="Multi-Turn Conversations",
    ),
    TestCase(
        name="Ambiguity: 'compare those' is never resolvable regardless of context",
        turns=[
            "Tell me about CP312.",
            "Compare those.",
        ],
        check=lambda resp, at: is_clarification_response(resp),
        category="Multi-Turn Conversations",
        metric="clarification",
    ),
]

# Protects: the entity-history model itself (Sprint 9B) - ordinal
# references and list resolution, which had NO resolution mechanism at
# all before this sprint (always clarified, since the four-slot memory
# had no concept of a shown list or a position within one). Exercised
# end-to-end via the real chatbot, not direct structured_search() calls.
ENTITY_HISTORY_TESTS = [
    TestCase(
        name="Ordinal resolution: 'the first one' after a department faculty list",
        turns=[
            "Who works in Marketing?",
            "Tell me about the first one.",
        ],
        check=lambda resp, at: (
            contains_any(resp, "Ammara Mahmood")
            and not is_clarification_response(resp)
        ),
        category="Entity History",
    ),
    TestCase(
        name="Ordinal resolution: 'the second one' resolves a DIFFERENT person from the same list",
        turns=[
            "Who works in Marketing?",
            "Tell me about the second one.",
        ],
        check=lambda resp, at: (
            contains_any(resp, "Chatura Ranaweera")
            and not is_clarification_response(resp)
        ),
        category="Entity History",
    ),
    TestCase(
        name="Multi-hop ordinal: course -> prerequisite list -> 'the first one'",
        turns=[
            "Tell me about CP312.",
            "What are its prerequisites?",
            "Tell me about the first one.",
        ],
        check=lambda resp, at: (
            contains_any(resp, "CP264")
            and not is_clarification_response(resp)
        ),
        category="Entity History",
    ),
    TestCase(
        name="List resolution: 'compare those' identifies real names instead of a bare generic decline",
        turns=[
            "Who works in Marketing?",
            "Compare those.",
        ],
        check=lambda resp, at: (
            is_clarification_response(resp)
            and contains_any(resp, "Mahmood", "Ranaweera")
        ),
        category="Entity History",
        metric="clarification",
    ),
    TestCase(
        name="Ordinal reference with no prior list still clarifies gracefully (no crash, no hallucination)",
        turns=[
            "Tell me about CP312.",
            "Tell me about the second one.",
        ],
        check=lambda resp, at: is_clarification_response(resp),
        category="Entity History",
        metric="clarification",
    ),
]

OUT_OF_DOMAIN_TESTS = [
    TestCase(
        name="Sports question",
        turns=["Tell me about the latest Super Bowl champion."],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
        metric="unsupported",
    ),
    TestCase(
        name="Celebrity / movie question",
        turns=["What's your favorite movie?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
        metric="unsupported",
    ),
    TestCase(
        name="Coding question",
        turns=["Can you write a Python function to sort a list?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
        metric="unsupported",
    ),
    TestCase(
        name="Politics question",
        turns=["Who is the president of the United States?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
        metric="unsupported",
    ),
    TestCase(
        name="General knowledge question",
        turns=["What's the weather like today?"],
        check=lambda resp, at: is_off_topic_decline(resp),
        category="Out-of-Domain Detection",
        metric="unsupported",
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
    ("Coordinator Lookup", "Coordinator", COORDINATOR_LOOKUP_TESTS),
    ("Department Coordinator", "Dept Coordinator", DEPARTMENT_COORDINATOR_TESTS),
    ("Course Prerequisites", "Prerequisites", COURSE_PREREQUISITE_TESTS),
    ("Course Metadata", "Course Metadata", COURSE_METADATA_TESTS),
    ("Graduate Program Requirements", "Grad Program Reqs", GRADUATE_PROGRAM_REQUIREMENTS_TESTS),
    ("Multi-Turn Conversations", "Multi-Turn", MULTI_TURN_CONVERSATION_TESTS),
    ("Entity History", "Entity History", ENTITY_HISTORY_TESTS),
    ("Out-of-Domain Detection", "Out-of-Domain", OUT_OF_DOMAIN_TESTS),
]

# Documentation (Sprint 8B requirement 6): every capability this project
# has shipped, and which evaluation category is responsible for
# protecting it against regression. Printed at the end of every run so
# coverage is visible without having to read this file.
CAPABILITY_COVERAGE = [
    ("Basic conversation / greetings", "Basic Conversation"),
    ("Conversation memory (follow-up phrases)", "Conversation Memory"),
    ("Program lookup (name/alias/acronym)", "Program Retrieval, Aliases, Comparison"),
    ("Single-person faculty lookup", "Faculty Retrieval"),
    ("Department lookup + single-word false-positive guard", "Faculty Retrieval, Department False-Positive Prevention"),
    ("Department -> faculty list / faculty-level list", "Faculty Retrieval"),
    ("Research-topic search (semantic, Chroma-backed)", "Research Topic"),
    ("Courses-taught lookup (direct, by code/name)", "Faculty Courses Taught"),
    ("Person + topic courses-taught lookup", "Person + Topic Courses Taught"),
    ("Program coordinator lookup", "Coordinator Lookup"),
    ("Department coordinator lookup (existing coordinator column, entity-history follow-up)", "Department Coordinator"),
    ("Course prerequisites (direct/reverse/relational)", "Course Prerequisites"),
    ("Course metadata exposure (corequisites/exclusions/location/notes)", "Course Metadata"),
    ("Graduate program requirements (direct/reverse/relational)", "Graduate Program Requirements"),
    ("Multi-turn contextual reference resolution", "Multi-Turn Conversations"),
    ("Entity-history ordinal/list resolution + cross-function write-back", "Entity History"),
    ("Out-of-domain detection", "Out-of-Domain Detection"),
    ("Scraper/extraction correctness (URLs, email, duplicates, prerequisites)", "Data Integrity"),
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


# Protects: the faculty-name encoding fix (Sprint 10C). Root cause was
# response.text decoding with requests' guessed encoding (falls back to
# ISO-8859-1 whenever the server's Content-Type omits a charset - WLU's
# pages do this, and are actually UTF-8), producing mojibake like
# "AngÃ¨le Foley" for "Angèle Foley". Fixed at the source
# (get_faculty_links.py/load_faculty.py now parse response.content, not
# response.text) and the 17 previously-corrupted rows were rebuilt via a
# targeted re-fetch (not hardcoded string replacement). Pass criteria:
# every specific name named in the fix is now exactly correct, and no
# mojibake byte-pattern remains anywhere in the table (a general sweep,
# not just the known cases - guards against any row this investigation
# missed and against the bug ever recurring).
def _check_faculty_name_encoding():

    import sqlite3

    conn = sqlite3.connect(str(ROOT_DIR / "data" / "faculty.db"))
    cursor = conn.cursor()

    # Specific, previously-confirmed-corrupted names (the exact
    # before/after cases named in this fix), checked for exact equality
    # against their correct, real spelling.
    known_corrections = [
        ("angele-foley", "Angèle Foley"),
        ("magnus-mfoafo-mcarthy", "Magnus Mfoafo-M’Carthy"),
        ("bill-oleary", "William O’Leary"),
        ("micheal-j-kelly", "Micheál J. Kelly"),
        ("karljurgen-feuerherm", "Karljürgen Feuerherm"),
        ("jorg-broschek", "Jörg Broschek"),
        ("john-edison-munoz-cardona", "John Edison Muñoz Cardona"),
        ("amy-clements-cortes", "Amy Clements-Cortés"),
        ("dirk-wallschlager", "Dirk Wallschläger"),
        ("renee-s-macphee", "Renée S. MacPhee"),
        ("ginette-lafreniere", "Ginette Lafrenière"),
        ("david-rose", "David Rosé"),
    ]

    for url_fragment, expected_name in known_corrections:

        cursor.execute(
            "SELECT name FROM faculty WHERE source_url LIKE ?",
            (f"%/{url_fragment}/%",)
        )
        row = cursor.fetchone()

        yield (
            f"Faculty name encoding: {url_fragment} is stored correctly",
            bool(row) and row[0] == expected_name,
            f"got {row[0] if row else None!r}, expected {expected_name!r}"
        )

    # General sweep: no row's name/title/research_interests/biography
    # anywhere in the table still exhibits the mojibake pattern (raw
    # UTF-8 bytes mis-decoded as Latin-1) - a round-trip re-encode/decode
    # that succeeds AND changes the text is the reliable signature (a
    # bare "Ã" or "â" alone is NOT reliable on its own, since both are
    # also legitimate standalone characters in genuinely-correct
    # non-English text - e.g. "Fordlândia" in one real bio - which this
    # sweep deliberately does not flag).
    def _is_mojibake(text):
        if not text:
            return False
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False
        return repaired != text

    cursor.execute("SELECT name, title, research_interests, biography, source_url FROM faculty")
    rows = cursor.fetchall()

    remaining = [
        (name, field, url)
        for name, title, research, bio, url in rows
        for field, val in [("name", name), ("title", title), ("research_interests", research), ("biography", bio)]
        if _is_mojibake(val)
    ]

    conn.close()

    yield (
        "Faculty name encoding: no mojibake remains anywhere in faculty.db",
        len(remaining) == 0,
        f"found {len(remaining)} still-corrupted fields: {remaining[:5]!r}" if remaining else ""
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


# Protects: exposure of existing courses.db metadata fields (Sprint 10D)
# - prerequisites_text, corequisites_text, exclusions_text,
# location_text, notes_text were already scraped and stored but never
# shown in a plain course lookup's context before this sprint. Purely
# additive: no new search function, no new intent pattern, no routing
# change - just more of the same already-fetched row surfaced. Checked
# via direct structured_search() calls (not AppTest) since the exact
# field labels and their order in the raw context text is what's being
# verified, not the LLM's paraphrase of it. Pass criteria: each field
# appears, labeled, only when actually populated, in the fixed order
# Prerequisites/Corequisites/Exclusions/Location/Notes, and a course
# with none of the five renders identically to before this sprint.
def _check_course_metadata_exposure():

    sys.path.insert(0, str(SRC_DIR))
    from retriever import structured_search

    # CP312: prerequisites only (already verified elsewhere in this
    # suite to have real, specific prerequisite text).
    result = structured_search("What is CP312?", {})
    text = result[0] if result else ""
    yield (
        "Course metadata: CP312's plain lookup now includes Prerequisites",
        bool(result) and "Prerequisites: CP264" in text,
        f"got {text!r}"
    )
    yield (
        "Course metadata: CP312 correctly omits Corequisites/Exclusions/Location/Notes (not populated)",
        bool(result) and not any(
            label in text for label in ("Corequisites:", "Exclusions:", "Location:", "Notes:")
        ),
        f"got {text!r}"
    )

    # EN645: corequisites only.
    result = structured_search("What is EN645?", {})
    text = result[0] if result else ""
    yield (
        "Course metadata: EN645's plain lookup includes Corequisites",
        bool(result) and "Corequisites:" in text,
        f"got {text!r}"
    )

    # UU400: prerequisites + exclusions + location, in that fixed order,
    # with corequisites/notes correctly absent.
    result = structured_search("What is UU400?", {})
    text = result[0] if result else ""
    has_all_three = bool(result) and all(
        label in text for label in ("Prerequisites:", "Exclusions:", "Location:")
    )
    correct_order = (
        has_all_three
        and text.find("Prerequisites:") < text.find("Exclusions:") < text.find("Location:")
    )
    yield (
        "Course metadata: UU400 includes Prerequisites, Exclusions, and Location in that order",
        correct_order,
        f"got {text!r}"
    )
    yield (
        "Course metadata: UU400 correctly omits Corequisites/Notes (not populated)",
        bool(result) and not any(label in text for label in ("Corequisites:", "Notes:")),
        f"got {text!r}"
    )

    # CQ609: exclusions + notes, in that fixed order.
    result = structured_search("What is CQ609?", {})
    text = result[0] if result else ""
    has_both = bool(result) and "Exclusions:" in text and "Notes:" in text
    yield (
        "Course metadata: CQ609 includes Exclusions and Notes in that order",
        has_both and text.find("Exclusions:") < text.find("Notes:"),
        f"got {text!r}"
    )

    # CS600: none of the five fields populated - context must render
    # identically to how it did before this sprint (no stray blank
    # metadata section), while all pre-existing course information
    # (code/name/credits/department/description) is still present.
    result = structured_search("What is CS600?", {})
    text = result[0] if result else ""
    yield (
        "Course metadata: CS600 (no metadata fields) shows no metadata section at all",
        bool(result) and not any(
            label in text
            for label in ("Prerequisites:", "Corequisites:", "Exclusions:", "Location:", "Notes:")
        ),
        f"got {text!r}"
    )
    yield (
        "Course metadata: CS600's existing course information is fully preserved",
        bool(result) and all(
            field in text
            for field in ("Course Code: CS600", "Credits:", "Department:", "Description:")
        ),
        f"got {text!r}"
    )


# Protects: direct department coordinator lookup (Sprint 10E), reading
# departments.db's existing `coordinator` column only - never inferred
# from free-text description. Checked via direct structured_search()/
# resolve_contextual_reference() calls (not AppTest) since the exact
# presence/absence of the "Department Coordinator:" section and its
# grounding data is what's being verified, not the LLM's paraphrase.
def _check_department_coordinator():

    sys.path.insert(0, str(SRC_DIR))
    from retriever import structured_search, resolve_contextual_reference, create_memory

    # Direct queries, departments known to have real coordinator data.
    result = structured_search("Who coordinates the History department?", {})
    text = result[0] if result else ""
    yield (
        "Department coordinator: History resolves to the real coordinator",
        bool(result) and "Department Coordinator:" in text and "Susan Neylan" in text,
        f"got {text[:150]!r}"
    )

    # Biology is a single-word department name with no other academic-
    # signal word in this phrasing - specifically verifies the
    # "coordinat*" signal-word addition (Sprint 10E) that makes this
    # resolvable at all.
    result = structured_search("Who is the coordinator of Biology?", {})
    text = result[0] if result else ""
    yield (
        "Department coordinator: single-word department name (Biology) resolves correctly",
        bool(result) and "Department Coordinator:" in text and "Jonathan Wilson" in text,
        f"got {text[:150]!r}"
    )

    result = structured_search("Department coordinator for Economics", {})
    text = result[0] if result else ""
    yield (
        "Department coordinator: Economics resolves to the real coordinator",
        bool(result) and "Department Coordinator:" in text and "Christos Shiamptanis" in text,
        f"got {text[:150]!r}"
    )

    # Department without coordinator data - graceful "not available",
    # never a fabricated name.
    result = structured_search(
        "Who coordinates the Archaeology and Heritage Studies department?", {}
    )
    text = result[0] if result else ""
    yield (
        "Department coordinator: department without data gets a graceful fallback, not a fabrication",
        bool(result) and "Coordinator information is not available." in text,
        f"got {text[:150]!r}"
    )

    # Unsupported department - doesn't exist at all - must never
    # fabricate a department or a coordinator for it.
    result = structured_search(
        "Who coordinates the Underwater Basketweaving department?", {}
    )
    yield (
        "Department coordinator: nonexistent department produces no fabricated result",
        result is None,
        f"got {result!r}"
    )

    # Plain department query with no coordinator intent - must show no
    # coordinator section at all (identical to pre-Sprint-10E output).
    result = structured_search("Tell me about the History department", {})
    text = result[0] if result else ""
    yield (
        "Department coordinator: plain department query omits the coordinator section entirely",
        bool(result) and "Department Coordinator:" not in text,
        f"got {text[:150]!r}"
    )

    # Multi-turn: "who coordinates it?" after establishing department
    # context via entity-history, with no legacy-slot involvement.
    mem = create_memory()
    mem["turn_count"] = 1
    structured_search("Tell me about the History department", mem)
    mem["turn_count"] = 2
    result = resolve_contextual_reference("Who coordinates it?", mem)
    yield (
        "Department coordinator: multi-turn 'who coordinates it?' resolves via entity-history",
        bool(result) and result[0] == "resolved" and "Susan Neylan" in result[1],
        f"got {result[0] if result else None!r}"
    )

    # Multi-turn: the task's own literal example ("tell me more about
    # them") - no "coordinat" keyword at all, so this exercises the
    # generic type-priority substitution path (with the "department"
    # qualifier fix), not the dedicated coordinator rewrite.
    mem = create_memory()
    mem["turn_count"] = 1
    structured_search("Tell me about the History department", mem)
    mem["turn_count"] = 2
    result = resolve_contextual_reference("Tell me more about them.", mem)
    yield (
        "Department coordinator: 'tell me more about them' still resolves the department (single-word name)",
        bool(result) and result[0] == "resolved" and "History" in result[1],
        f"got {result[0] if result else None!r}"
    )

    # Recency disambiguation: program established, then department -
    # "who coordinates it?" must follow the department (more recent).
    mem = create_memory()
    mem["turn_count"] = 1
    structured_search("What is the Master of Applied Computing program?", mem)
    mem["turn_count"] = 2
    structured_search("Tell me about the History department", mem)
    mem["turn_count"] = 3
    result = resolve_contextual_reference("Who coordinates it?", mem)
    yield (
        "Department coordinator: recency disambiguation follows the department when it's more recent",
        bool(result) and result[0] == "resolved" and "Susan Neylan" in result[1],
        f"got {result[0] if result else None!r}"
    )

    # Recency disambiguation, reversed - department established, then
    # program - "who coordinates it?" must follow the program instead,
    # confirming the existing program-coordinator capability still works
    # unchanged through the new dynamic resolution.
    mem = create_memory()
    mem["turn_count"] = 1
    structured_search("Tell me about the History department", mem)
    mem["turn_count"] = 2
    structured_search("What is the Master of Applied Computing program?", mem)
    mem["turn_count"] = 3
    result = resolve_contextual_reference("Who coordinates it?", mem)
    yield (
        "Department coordinator: recency disambiguation follows the program when it's more recent",
        bool(result) and result[0] == "resolved"
        and any(name in result[1] for name in ("Dariush Ebrahimi", "Usama Mir")),
        f"got {result[0] if result else None!r}"
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


def _check_entity_history_and_writeback():

    sys.path.insert(0, str(SRC_DIR))
    from retriever import (
        create_memory, search_course, search_faculty_by_department,
        structured_search, resolve_contextual_reference,
    )

    # Schema: the legacy four-slot dict stays exactly as it was (Sprint
    # 9B requirement - do not remove it yet), with entity_history added
    # alongside it, not in place of it.
    mem = create_memory()

    yield (
        "create_memory() keeps all four legacy memory slots",
        all(
            k in mem
            for k in ("last_course", "last_program", "last_department", "last_faculty")
        ),
        f"got keys: {sorted(mem.keys())}"
    )

    yield (
        "create_memory() adds an empty entity_history alongside the legacy slots",
        "entity_history" in mem and len(mem["entity_history"]) == 0,
        f"got {mem.get('entity_history')!r}"
    )

    # Write-back: a function that already wrote the legacy slot before
    # Sprint 9B (search_course) now writes entity_history too, without
    # changing its legacy write at all.
    mem = create_memory()
    search_course("What is CP312?", mem)

    yield (
        "Write-back: search_course populates both the legacy slot and entity_history",
        mem["last_course"] == "CP312"
        and any(
            e["entity_type"] == "course" and e["entity_id"] == "CP312"
            for e in mem["entity_history"]
        ),
        f"last_course={mem['last_course']!r}, "
        f"entity_history={list(mem['entity_history'])!r}"
    )

    # Write-back: search_faculty_by_department was one of the functions
    # Sprint 9A found writes NOTHING at all - it has no legacy slot to
    # write to, so this only shows up in entity_history.
    mem = create_memory()
    search_faculty_by_department("Who works in Marketing?", mem)
    faculty_entries = [
        e for e in mem["entity_history"] if e["entity_type"] == "faculty"
    ]

    yield (
        "Write-back: search_faculty_by_department (previously silent) now populates entity_history as a list",
        len(faculty_entries) >= 2 and all(e["list_id"] for e in faculty_entries),
        f"got {len(faculty_entries)} faculty entries, "
        f"list_ids={ {e['list_id'] for e in faculty_entries} }"
    )

    # Write-back: search_faculty_courses_taught was another previously-
    # silent function - closing this exact gap is what lets "who teaches
    # it?" -> "does that professor..." resolve instead of clarifying
    # (see the updated Multi-Turn Conversations test).
    mem = create_memory()
    structured_search("Who has taught CP312?", mem)
    course_taught_faculty = [
        e for e in mem["entity_history"]
        if e["entity_type"] == "faculty"
        and e["source_function"] == "search_faculty_courses_taught"
    ]

    yield (
        "Write-back: search_faculty_courses_taught (previously silent) now populates entity_history",
        len(course_taught_faculty) >= 1,
        f"got {course_taught_faculty!r}"
    )

    # Ordinal resolution: real data, real positions - "the first one" and
    # "the second one" must resolve to two DIFFERENT real people from the
    # same list, not the same person twice or an arbitrary guess.
    mem_first = create_memory()
    structured_search("Who works in Marketing?", mem_first)
    first = resolve_contextual_reference("Tell me about the first one.", mem_first)

    mem_second = create_memory()
    structured_search("Who works in Marketing?", mem_second)
    second = resolve_contextual_reference("Tell me about the second one.", mem_second)

    yield (
        "Ordinal resolution: 'first' and 'second' resolve to different real people from the same list",
        bool(first) and bool(second)
        and first[0] == "resolved" and second[0] == "resolved"
        and first[1] != second[1],
        f"first={first[1][:60] if first and len(first) > 1 else first!r}, "
        f"second={second[1][:60] if second and len(second) > 1 else second!r}"
    )

    # Ordinal resolution: no list has ever been shown in this
    # conversation - must clarify gracefully, never crash or invent a
    # position that doesn't exist.
    mem = create_memory()
    search_course("What is CP312?", mem)
    result = resolve_contextual_reference("Tell me about the second one.", mem)

    yield (
        "Ordinal resolution: no prior list yields a graceful clarification, not a crash",
        bool(result) and result[0] == "clarify",
        f"got {result!r}"
    )

    # List resolution: "compare those" is still a capability gap (no
    # comparison feature exists for any entity type), but now identifies
    # the real resolved entities by name in its clarification message
    # instead of the bare generic decline used before Sprint 9B.
    mem = create_memory()
    structured_search("Who works in Marketing?", mem)
    result = resolve_contextual_reference("Compare those.", mem)

    yield (
        "List resolution: 'compare those' clarifies with real identified names, not the bare generic message",
        bool(result) and result[0] == "clarify" and result[1] != (
            "I'm not sure what you're referring to. Could you clarify "
            "or provide a bit more detail?"
        ),
        f"got {result!r}"
    )


def run_data_integrity_checks():

    checks = list(_check_url_normalization_collapses_variants())
    checks += list(_check_no_duplicate_faculty_records())
    checks += list(_check_email_extraction())
    checks += list(_check_course_prerequisites())
    checks += list(_check_course_metadata_exposure())
    checks += list(_check_department_coordinator())
    checks += list(_check_image_link_rejection())
    checks += list(_check_beggar_recovered())
    checks += list(_check_contextual_reference_resolution())
    checks += list(_check_program_course_requirements())
    checks += list(_check_entity_history_and_writeback())
    checks += list(_check_faculty_name_encoding())

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

    # Sprint 8B reporting: pass/fail rolled up by TestCase.metric, across
    # every category, independent of which capability produced it.
    metric_scores = {"retrieval": [0, 0], "clarification": [0, 0], "unsupported": [0, 0]}

    for category, _, tests in CATEGORIES:
        passed_count = 0

        for test in tests:
            passed = run_test(test)

            if passed:
                passed_count += 1

            bucket = metric_scores.setdefault(test.metric, [0, 0])
            bucket[1] += 1

            if passed:
                bucket[0] += 1

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

    print("\n" + "=" * 36)
    print("Capability Coverage")
    print("=" * 36)
    print(
        "(every shipped capability and the category that protects it "
        "against regression)\n"
    )

    for capability, category_label in CAPABILITY_COVERAGE:
        print(f"  - {capability}")
        print(f"      -> {category_label}")

    print("\n" + "=" * 36)
    print("Cross-Cutting Accuracy Metrics")
    print("=" * 36 + "\n")

    retrieval_passed, retrieval_total = metric_scores["retrieval"]
    clarification_passed, clarification_total = metric_scores["clarification"]
    unsupported_passed, unsupported_total = metric_scores["unsupported"]

    print(
        f"Retrieval accuracy "
        f"(did the right feature answer with correct data): "
        f"{retrieval_passed}/{retrieval_total}"
    )
    print(
        f"Clarification accuracy "
        f"(unresolvable references correctly ask for clarification): "
        f"{clarification_passed}/{clarification_total}"
    )
    print(
        f"Unsupported-query handling "
        f"(out-of-scope questions decline gracefully, never fabricate): "
        f"{unsupported_passed}/{unsupported_total}"
    )


if __name__ == "__main__":
    main()
