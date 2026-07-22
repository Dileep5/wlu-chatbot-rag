import os
import sqlite3
import re
from collections import deque
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/vector_db"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(
    path=DB_DIR
)

collection = client.get_collection(
    "wlu_chatbot_chunks"
)

# The faculty-research collection is built by the separate
# build_faculty_vector_db.py pipeline (Sprint 4C1) and may not exist yet
# in every environment - detected once, the same way FACULTY_DB_READY
# guards faculty.db access, so its absence degrades gracefully instead of
# raising.
try:
    faculty_research_collection = client.get_collection(
        "wlu_faculty_research"
    )
    FACULTY_RESEARCH_READY = True
except Exception:
    faculty_research_collection = None
    FACULTY_RESEARCH_READY = False

FOLLOWUP_PHRASES = [
    "tell me more",
    "more",
    "explain",
    "details",
    "more details",
    "what about this",
    "what about it"
]

# Trailing punctuation a user might naturally type after a follow-up
# phrase ("Tell me more.", "More?", "Explain!") - stripped before checking
# membership in FOLLOWUP_PHRASES so punctuation doesn't defeat an exact
# match. Used everywhere that membership check happens.
_FOLLOWUP_TRAILING_PUNCTUATION = ".,!?;: "


def normalize_followup_text(text):

    return text.lower().strip().rstrip(_FOLLOWUP_TRAILING_PUNCTUATION)


# -----------------------------
# Entity history (Sprint 9B)
#
# Introduced alongside the original four-slot memory (last_course/
# last_program/last_department/last_faculty), which stays exactly as it
# was - every existing read/write site for those four keys is untouched.
# entity_history is a bounded, ordered log of every entity any retrieval
# function has surfaced, richer than a single scalar per type: it carries
# *which* function produced it, *when* (turn_number), how confidently,
# and - for list-shaped answers (a department's faculty, a reverse
# prerequisite lookup) - its position within that list, which is what
# lets ordinal references ("the second one") resolve at all. The four-
# slot dict has no way to represent any of this.
# Deliberately larger than a single list write's cap (see
# _record_entity_list's max_entries below) - a list write followed by a
# trailing "primary subject" write (the ordering convention used
# throughout this file) must never fill the deque so completely that the
# trailing write evicts the list's own earliest position and corrupts
# ordinal resolution for it.
ENTITY_HISTORY_SIZE = 12

# Every entity type an entity-history entry can carry. Mirrors the four
# legacy types (course/program/department/faculty) plus two new ones
# (faculty_institution for "Faculty of Science"-style lookups) that never
# had a legacy slot to begin with - see _resolve_typed_value(), which
# falls back to the legacy dict only for types that have one.
_MEMORY_KEY_BY_TYPE = {
    "course": "last_course",
    "program": "last_program",
    "department": "last_department",
    "faculty": "last_faculty",
}


def create_memory():
    """Fresh session memory: the original four-slot dict, unchanged, plus
    the new entity-history structures alongside it (Sprint 9B). Nothing
    reads this function's return value differently than a plain literal
    dict - it just keeps the schema in one place."""

    return {
        "last_course": None,
        "last_program": None,
        "last_department": None,
        "last_faculty": None,
        "turn_count": 0,
        "entity_history": deque(maxlen=ENTITY_HISTORY_SIZE),
        "_list_counter": 0,
        "_last_list_id": None,
    }


def _record_entity(
    memory, entity_type, entity_id, display_name, source_function,
    confidence="exact", list_id=None, list_position=None
):
    """Append one entity to memory['entity_history']. A no-op when memory
    is None (mirrors the existing 'if memory is not None' guard used by
    every legacy write site) and safe against a plain dict that doesn't
    already have an 'entity_history' key (older/manually-built memory
    dicts, e.g. in tests)."""

    if memory is None or not entity_id:
        return

    history = memory.setdefault(
        "entity_history", deque(maxlen=ENTITY_HISTORY_SIZE)
    )

    history.append({
        "entity_type": entity_type,
        "entity_id": entity_id,
        "display_name": display_name,
        "source_function": source_function,
        "turn_number": memory.get("turn_count", 0),
        "confidence": confidence,
        "list_id": list_id,
        "list_position": list_position,
    })


def _record_entity_list(memory, entity_type, entities, source_function, max_entries=5):
    """Record a list-shaped result (e.g. a department's faculty, a
    reverse prerequisite lookup) as multiple entity-history entries
    sharing one list_id, in display order. 'entities' is an iterable of
    (entity_id, display_name) pairs. Also updates memory['_last_list_id']
    so ordinal references ("the second one") always resolve against the
    MOST RECENT list, independent of entity-history scan order.

    Capped at 5 (not ENTITY_HISTORY_SIZE) deliberately: ordinal support
    only recognizes first..fifth/last anyway (see _ORDINAL_POSITIONS),
    and keeping list writes well under the deque's capacity leaves room
    for a trailing primary-subject write (e.g. the course a courses-
    taught list answers) without evicting the list's own early
    positions."""

    if memory is None:
        return None

    entities = [e for e in entities if e[0]]

    if not entities:
        return None

    memory["_list_counter"] = memory.get("_list_counter", 0) + 1
    list_id = f"L{memory['_list_counter']}_{source_function}"

    for position, (entity_id, display_name) in enumerate(
        entities[:max_entries], start=1
    ):
        _record_entity(
            memory, entity_type, entity_id, display_name, source_function,
            confidence="inferred" if len(entities) > 1 else "exact",
            list_id=list_id, list_position=position,
        )

    memory["_last_list_id"] = list_id

    return list_id


def _latest_entity_of_type(memory, entity_type):
    """The single best entity of a given type to use for a bare pronoun
    ("it", "that professor"). Restricted to the most recent turn any
    entity of this type was recorded in; within that turn, a standalone
    entity (list_id is None - the turn's primary subject) is preferred
    over a list entry, and among list entries the first-listed
    (list_position 1) is preferred - a reasonable default when multiple
    equally-recent candidates exist (e.g. two instructors returned by one
    "who has taught" lookup) and there's no stronger signal to prefer one
    over the other."""

    if memory is None:
        return None

    history = memory.get("entity_history")

    if not history:
        return None

    matches = [e for e in history if e["entity_type"] == entity_type]

    if not matches:
        return None

    max_turn = max(e["turn_number"] for e in matches)
    candidates = [e for e in matches if e["turn_number"] == max_turn]

    candidates.sort(
        key=lambda e: (e["list_id"] is not None, e.get("list_position") or 0)
    )

    return candidates[0]


def _resolve_typed_value(memory, entity_type):
    """The value resolve_contextual_reference() should substitute for a
    given entity type - entity_history first (richer, covers every
    Sprint 4-8 capability), falling back to the legacy four-slot dict
    only for types that have a legacy slot at all. This is what lets
    resolution keep working unchanged for memory dicts built the old way
    (a plain {'last_course': 'CP312'} literal, as several existing tests
    still do) while picking up richer history when it's present.

    Returns the entry's display_name, not its entity_id: for
    course/program/department the two are normally the same searchable
    text (course code, program name, department name), but for faculty
    entity_id is a stable source_url (the Sprint 6B convention, useful
    for dedup/joins) that search_faculty()'s name-matching tiers can't
    match against - only display_name (the person's actual name) is
    valid text to substitute back into a question."""

    if memory is None:
        return None

    entry = _latest_entity_of_type(memory, entity_type)

    if entry:
        return entry["display_name"]

    legacy_key = _MEMORY_KEY_BY_TYPE.get(entity_type)

    return memory.get(legacy_key) if legacy_key else None


def _entities_in_list(memory, list_id):

    if memory is None or not list_id:
        return []

    history = memory.get("entity_history") or []

    return sorted(
        (e for e in history if e.get("list_id") == list_id),
        key=lambda e: e["list_position"]
    )


def _table_has_level_column(db_path, table_name):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    return "level" in columns


# The undergraduate-support sprint adds a `level` column to courses/programs/
# departments, but older databases (before that migration has been re-run)
# won't have it yet. Detect this once so retrieval keeps working either way.
COURSES_HAVE_LEVEL = _table_has_level_column("data/courses.db", "courses")
PROGRAMS_HAVE_LEVEL = _table_has_level_column("data/programs.db", "programs")
DEPARTMENTS_HAVE_LEVEL = _table_has_level_column("data/departments.db", "departments")


def _table_exists(db_path, table_name):

    if not os.path.exists(db_path):
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    exists = cursor.fetchone() is not None
    conn.close()

    return exists


# faculty.db doesn't exist at all until the faculty scrape has been run.
# Detect this so search_faculty can no-op gracefully rather than raising
# "no such table" and breaking every query that reaches structured_search.
FACULTY_DB_READY = _table_exists("data/faculty.db", "faculty")

# faculty_courses_taught is a separate, newer table - may not exist yet
# in every environment even when faculty.db itself is ready.
FACULTY_COURSES_TAUGHT_READY = _table_exists(
    "data/faculty.db", "faculty_courses_taught"
)

# course_prerequisite_refs is a separate, newer table (Sprint 6F) - may
# not exist yet in every environment even when courses.db itself is
# ready.
COURSE_PREREQUISITE_REFS_READY = _table_exists(
    "data/courses.db", "course_prerequisite_refs"
)

# program_course_requirements is a separate, newer table (Sprint 7D),
# graduate-only by design - may not exist yet in every environment even
# when programs.db itself is ready.
PROGRAM_COURSE_REQUIREMENTS_READY = _table_exists(
    "data/programs.db", "program_course_requirements"
)


def search_course(question, memory=None):

    course_match = re.search(
        r"\b[A-Z]{2,4}\d{3}[A-Z]?\b",
        question.upper()
    )

    if not course_match:
        return None

    course_code = course_match.group()

    conn = sqlite3.connect(
        "data/courses.db"
    )

    # Sprint 10D: the additional metadata fields below are appended
    # after the conditionally-present `level` column, so their
    # positional index would otherwise depend on COURSES_HAVE_LEVEL
    # (fragile, easy to miscount). sqlite3.Row supports the exact same
    # positional access every existing caller already uses (result[0],
    # result[1], ...) while also allowing safe name-based access
    # (result["exclusions_text"]) for the new fields regardless of
    # where `level` did or didn't shift things.
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    level_column = ", level" if COURSES_HAVE_LEVEL else ""

    cursor.execute(f"""
    SELECT
        course_code,
        course_name,
        credits,
        description,
        department_name,
        source_url
        {level_column},
        prerequisites_text,
        corequisites_text,
        exclusions_text,
        location_text,
        notes_text
    FROM courses
    WHERE course_code=?
    """, (course_code,))

    result = cursor.fetchone()

    conn.close()

    if result and memory is not None:
        memory["last_course"] = course_code
        _record_entity(
            memory, "course", course_code, f"{course_code} - {result[1]}",
            "search_course",
        )

    return result


_COURSE_CODE_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")

# Four deterministic patterns, tried in this order (most specific first)
# so none of them can shadow another:
#   1. reverse lookup   - "which courses require CP264?"
#   2. no-prerequisite  - "what courses have no prerequisites listed?"
#   3. relational       - "does CP312 require CP220?"
#   4. direct lookup    - "what are the prerequisites for CP600?"
# All four require course-code-shaped tokens in the right place, so none
# of them collide with the "who has taught" / "has X taught Y courses"
# detectors above (which require the word "taught", never present here).
_REVERSE_PREREQUISITE_PATTERN = re.compile(
    r"\bwhich\s+courses?\s+requires?\s+(.+)", re.IGNORECASE
)
_NO_PREREQUISITE_PATTERN = re.compile(
    r"\bwhat\s+courses?\s+(?:have|has)\s+no\s+prerequisites?\b",
    re.IGNORECASE
)
_REQUIRES_RELATIONSHIP_PATTERN = re.compile(
    r"\bdoes\s+(.+?)\s+requires?\s+(.+)", re.IGNORECASE
)
_DIRECT_PREREQUISITE_PATTERN = re.compile(
    r"\bprerequisites?\s+(?:for|of)\s+(.+)", re.IGNORECASE
)


def _extract_course_code(text):

    match = _COURSE_CODE_TOKEN_PATTERN.search(text.upper())

    return match.group() if match else None


def _handle_direct_prerequisite_lookup(captured, memory=None):

    course_code = _extract_course_code(captured)

    if not course_code:
        return None

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT course_name, prerequisites_text FROM courses "
        "WHERE course_code = ?",
        (course_code,)
    )

    row = cursor.fetchone()

    conn.close()

    if not row:
        return (f"No course named {course_code} was found.", None)

    course_name, prerequisites_text = row

    # The listed prerequisite courses themselves become resolvable
    # entities too (e.g. "tell me about the first one" after "what are
    # CP312's prerequisites?") - written before the subject course below
    # so the subject stays the most recent single entity for bare
    # pronoun resolution ("it" should still mean CP312, not its last
    # prerequisite).
    if prerequisites_text:
        prereq_codes = list(dict.fromkeys(
            _COURSE_CODE_TOKEN_PATTERN.findall(prerequisites_text.upper())
        ))
        if prereq_codes:
            _record_entity_list(
                memory, "course",
                [(code, code) for code in prereq_codes],
                "search_course_prerequisites",
            )

    _record_entity(
        memory, "course", course_code, f"{course_code} - {course_name}",
        "search_course_prerequisites",
    )

    if not prerequisites_text:
        return (
            f"No prerequisites are listed for {course_code} "
            f"({course_name}).",
            None
        )

    return (
        f"Prerequisites for {course_code} ({course_name}):\n"
        f"{prerequisites_text}",
        None
    )


def _handle_reverse_prerequisite_lookup(captured, memory=None):

    required_code = _extract_course_code(captured)

    if not required_code:
        return None

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT course_code FROM course_prerequisite_refs "
        "WHERE required_course_code = ? ORDER BY course_code",
        (required_code,)
    )

    codes = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not codes:
        _record_entity(
            memory, "course", required_code, required_code,
            "search_course_prerequisites",
        )
        return (
            f"No courses were found that list {required_code} as a "
            f"prerequisite.",
            None
        )

    _record_entity_list(
        memory, "course", [(code, code) for code in codes],
        "search_course_prerequisites",
    )
    _record_entity(
        memory, "course", required_code, required_code,
        "search_course_prerequisites",
    )

    total = len(codes)
    displayed = codes[:25]

    lines = "\n".join(f"- {code}" for code in displayed)

    truncation = (
        f"\n(Showing {len(displayed)} of {total} courses.)"
        if total > len(displayed) else ""
    )

    return (
        f"Courses that require {required_code} as a prerequisite:\n"
        f"{lines}{truncation}",
        None
    )


def _handle_requires_relationship(course_phrase, required_phrase, memory=None):

    course_code = _extract_course_code(course_phrase)
    required_code = _extract_course_code(required_phrase)

    if not course_code or not required_code:
        return None

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM course_prerequisite_refs "
        "WHERE course_code = ? AND required_course_code = ?",
        (course_code, required_code)
    )

    has_ref = cursor.fetchone() is not None

    cursor.execute(
        "SELECT prerequisites_text FROM courses WHERE course_code = ?",
        (course_code,)
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return (f"No course named {course_code} was found.", None)

    prerequisites_text = row[0]

    _record_entity(
        memory, "course", course_code, course_code,
        "search_course_prerequisites",
    )

    # The derived reference table is checked first (fast, exact), but a
    # direct word-boundary check against the raw prerequisites_text is
    # still tried as a fallback - the reference table is a best-effort
    # extraction and can miss complex phrasing the raw text still states
    # plainly.
    text_confirms = bool(
        prerequisites_text
        and re.search(rf"\b{required_code}\b", prerequisites_text.upper())
    )

    if has_ref or text_confirms:
        return (
            f"Yes, {course_code} lists {required_code} as a "
            f"prerequisite.",
            None
        )

    if prerequisites_text:
        return (
            f"{required_code} is not listed as a prerequisite for "
            f"{course_code}. {course_code}'s listed prerequisites: "
            f"{prerequisites_text}",
            None
        )

    return (
        f"No prerequisites are listed for {course_code}, so "
        f"{required_code} is not a listed requirement.",
        None
    )


def _handle_no_prerequisite_courses(memory=None):

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT course_code FROM courses "
        "WHERE prerequisites_text IS NULL OR TRIM(prerequisites_text) = '' "
        "ORDER BY course_code"
    )

    codes = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not codes:
        return ("Every course has a prerequisite listed.", None)

    _record_entity_list(
        memory, "course", [(code, code) for code in codes],
        "search_course_prerequisites",
    )

    total = len(codes)
    displayed = codes[:25]

    lines = "\n".join(f"- {code}" for code in displayed)

    count_note = (
        f"Showing {len(displayed)} of {total} courses with no "
        f"prerequisite listed."
        if total > len(displayed) else
        "This reflects what's published for each course, not "
        "necessarily that none is required."
    )

    return (
        f"Courses with no prerequisite listed:\n{lines}\n"
        f"({count_note} This reflects what's published for each "
        f"course, not necessarily that none is required.)"
        if total > len(displayed) else
        f"Courses with no prerequisite listed:\n{lines}\n({count_note})",
        None
    )


def search_course_prerequisites(question, memory=None):

    if not COURSE_PREREQUISITE_REFS_READY:
        return None

    match = _REVERSE_PREREQUISITE_PATTERN.search(question)

    if match:
        return _handle_reverse_prerequisite_lookup(match.group(1), memory)

    if _NO_PREREQUISITE_PATTERN.search(question):
        return _handle_no_prerequisite_courses(memory)

    match = _REQUIRES_RELATIONSHIP_PATTERN.search(question)

    if match:
        return _handle_requires_relationship(
            match.group(1), match.group(2), memory
        )

    match = _DIRECT_PREREQUISITE_PATTERN.search(question)

    if match:
        return _handle_direct_prerequisite_lookup(match.group(1), memory)

    return None


# Graduate-only program-course requirement retrieval (Sprint 7D). The
# "does X require Y" shape is shared with _REQUIRES_RELATIONSHIP_PATTERN
# above, but that one only ever succeeds when BOTH sides look like course
# codes, and returns None (not a fallback) otherwise - so a query like
# "Does the Master of Applied Computing require CP600?" correctly falls
# through search_course_prerequisites() untouched and reaches this
# function next, which tries the same phrase shape with a program-name
# interpretation instead.
_REVERSE_PROGRAM_REQUIREMENT_PATTERN = re.compile(
    r"\bwhich\s+(?:graduate\s+)?programs?\s+requires?\s+(.+)", re.IGNORECASE
)
_PROGRAM_REQUIRES_COURSE_PATTERN = re.compile(
    r"\bdoes\s+(?:the\s+)?(.+?)\s+requires?\s+(.+)", re.IGNORECASE
)
_PROGRAM_REQUIRED_COURSES_PATTERN = re.compile(
    r"\brequired\s+courses?\b.*?\bfor\s+(?:the\s+)?(.+)", re.IGNORECASE
)


def _all_requirement_program_names():

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM program_course_requirements"
    )

    names = [row[0] for row in cursor.fetchall()]

    conn.close()

    return names


def _match_program_name(text):

    text_lower = text.lower()

    for name in sorted(_all_requirement_program_names(), key=len, reverse=True):

        if name.lower() in text_lower:
            return name

    return None


def _handle_reverse_program_requirement_lookup(captured, memory=None):

    course_code = _extract_course_code(captured)

    if not course_code:
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM program_course_requirements "
        "WHERE course_code = ? ORDER BY program_name",
        (course_code,)
    )

    programs = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not programs:
        _record_entity(
            memory, "course", course_code, course_code,
            "search_program_course_requirements",
        )
        return (
            f"No graduate program was found that lists {course_code} "
            f"as a required course, based on available structured data.",
            None
        )

    _record_entity_list(
        memory, "program", [(name, name) for name in programs],
        "search_program_course_requirements",
    )
    _record_entity(
        memory, "course", course_code, course_code,
        "search_program_course_requirements",
    )

    lines = "\n".join(f"- {name}" for name in programs)

    return (
        f"Graduate programs that require {course_code}:\n{lines}",
        None
    )


def _handle_program_requires_course(program_phrase, course_phrase, memory=None):

    course_code = _extract_course_code(course_phrase)

    if not course_code:
        return None

    program_name = _match_program_name(program_phrase)

    if not program_name:
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM program_course_requirements "
        "WHERE program_name = ? AND course_code = ?",
        (program_name, course_code)
    )

    found = cursor.fetchone() is not None

    conn.close()

    _record_entity(
        memory, "program", program_name, program_name,
        "search_program_course_requirements",
    )
    _record_entity(
        memory, "course", course_code, course_code,
        "search_program_course_requirements",
    )

    if found:
        return (
            f"Yes, {program_name} lists {course_code} as a required "
            f"course.",
            None
        )

    return (
        f"{course_code} is not listed as a required course for "
        f"{program_name}, based on available structured data. This "
        f"only covers explicitly required courses - electives and "
        f"categorical requirements aren't included.",
        None
    )


def _handle_program_required_courses(program_phrase, memory=None):

    program_name = _match_program_name(program_phrase)

    if not program_name:
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT course_code FROM program_course_requirements "
        "WHERE program_name = ? ORDER BY course_code",
        (program_name,)
    )

    codes = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not codes:
        _record_entity(
            memory, "program", program_name, program_name,
            "search_program_course_requirements",
        )
        return (
            f"No structured required-course data is available for "
            f"{program_name}.",
            None
        )

    _record_entity_list(
        memory, "course", [(code, code) for code in codes],
        "search_program_course_requirements",
    )
    _record_entity(
        memory, "program", program_name, program_name,
        "search_program_course_requirements",
    )

    lines = "\n".join(f"- {code}" for code in codes)

    return (
        f"Required courses listed for {program_name}:\n{lines}\n"
        f"(This reflects explicitly required courses only - electives "
        f"and categorical requirements, such as \"choose N credits "
        f"from...\", aren't included.)",
        None
    )


def search_program_course_requirements(question, memory=None):

    if not PROGRAM_COURSE_REQUIREMENTS_READY:
        return None

    match = _REVERSE_PROGRAM_REQUIREMENT_PATTERN.search(question)

    if match:
        result = _handle_reverse_program_requirement_lookup(
            match.group(1), memory
        )
        if result:
            return result

    match = _PROGRAM_REQUIRES_COURSE_PATTERN.search(question)

    if match:
        result = _handle_program_requires_course(
            match.group(1), match.group(2), memory
        )
        if result:
            return result

    match = _PROGRAM_REQUIRED_COURSES_PATTERN.search(question)

    if match:
        result = _handle_program_required_courses(match.group(1), memory)
        if result:
            return result

    return None


# Deterministic "who has taught" / "who teaches" intent detector -
# accepts past tense ("who taught X", "who has taught X") and present
# tense ("who teaches X") alike (Sprint 10C). Present-tense "teaches" is
# excluded via negative lookahead when immediately followed by "in"/
# "at"/"for" ("who teaches in Marketing?", "who teaches at the Business
# school?") - that specific shape is already claimed by the department-
# list detector below for a different meaning (a department/faculty
# list, not a single course's instructor), and is never a genuine course
# reference: no real course code or name is ever phrased as "in X"/
# "at X"/"for X" directly after "teaches". Kept as its own trigger,
# still never colliding with the department-list detector's "who
# teaches in/at" phrasing.
_TAUGHT_INTENT_PATTERN = re.compile(
    r"\bwho\s+(?:has\s+taught|taught|teaches(?!\s+(?:in|at|for)\b))\s+(.+)",
    re.IGNORECASE
)


# A bare pronoun/reference word captured on its own ("Who teaches it?",
# "Who has taught that?") is never a genuine course reference - it's a
# contextual follow-up meant for resolve_contextual_reference()'s memory-
# based substitution, called later in app.py's routing only once
# structured_search() has already returned None. Without this guard, the
# captured word falls through to search_faculty_courses_taught()'s
# course-name substring matching, which can wrongly match an unrelated
# course whose name simply contains the pronoun as a substring (e.g.
# "it" inside "Mobilities" - confirmed live, a Sprint 10C regression
# caught during verification and fixed here rather than shipped). This
# was always latently possible for "who taught it?" too (the "taught"
# tense alone had the same gap); adding "teaches" support is what
# actually exposed it via an existing test, so the guard is added for
# every tense uniformly rather than narrowly for the new one.
_BARE_REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those",
    "them", "they", "him", "her", "he", "she",
}


def _extract_taught_query(question):

    match = _TAUGHT_INTENT_PATTERN.search(question)

    if not match:
        return None

    captured = match.group(1).strip().rstrip("?.!, ")

    if captured.lower() in _BARE_REFERENCE_WORDS:
        return None

    return captured or None


def search_faculty_courses_taught(question, memory=None):

    if not FACULTY_COURSES_TAUGHT_READY:
        return None

    captured = _extract_taught_query(question)

    if not captured:
        return None

    # This query has clearly been recognized as a "who has taught" intent
    # from here on - every path below returns a real answer (including
    # graceful "not found" text) rather than None, so it can never fall
    # through to the general vector fallback and risk an ungrounded
    # answer for a course-specific question.

    code_match = re.search(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b", captured.upper())

    if code_match:
        course_code = code_match.group()
        label = course_code

    else:
        # No bare course code in the question - resolve a plain-English
        # course name (e.g. "Operating Systems") against courses.db first.
        conn = sqlite3.connect("data/courses.db")
        cursor = conn.cursor()
        cursor.execute("SELECT course_code, course_name FROM courses")
        course_rows = cursor.fetchall()
        conn.close()

        captured_lower = captured.lower()

        # Tier 1: exact name match. Tier 2: fallback substring match.
        course_code = None
        matched_name = None

        for code, course_name in course_rows:
            if course_name and course_name.strip().lower() == captured_lower:
                course_code, matched_name = code, course_name
                break

        if not course_code:
            for code, course_name in course_rows:
                if course_name and captured_lower in course_name.lower():
                    course_code, matched_name = code, course_name
                    break

        if not course_code:
            return (
                f'No course matching "{captured}" was found in the '
                f"course catalog.",
                None
            )

        label = f"{matched_name} ({course_code})"

    normalized_code = course_code.upper().replace(" ", "")

    conn = sqlite3.connect("data/faculty.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT faculty_source_url FROM faculty_courses_taught "
        "WHERE course_code = ?",
        (normalized_code,)
    )

    source_urls = [row[0] for row in cursor.fetchall()]

    if not source_urls:
        conn.close()
        _record_entity(
            memory, "course", normalized_code, label,
            "search_faculty_courses_taught",
        )
        return (
            f"No faculty-taught record was found for {label}. This "
            f"reflects faculty profiles' self-reported teaching history, "
            f"which isn't available for every course or instructor.",
            None
        )

    placeholders = ",".join("?" * len(source_urls))

    cursor.execute(
        f"SELECT DISTINCT name, title, source_url FROM faculty "
        f"WHERE source_url IN ({placeholders}) ORDER BY name",
        source_urls
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        _record_entity(
            memory, "course", normalized_code, label,
            "search_faculty_courses_taught",
        )
        return (
            f"No faculty-taught record was found for {label}.",
            None
        )

    # Recorded as a faculty list (using source_url - the stable key
    # convention established in Sprint 6B, not a name that could vary in
    # capitalization/credentials) so a later "that professor" can resolve
    # to whoever taught this course - the exact multi-hop gap (course ->
    # instructor -> ...) Sprint 9A identified as unresolvable before this
    # write-back existed. The subject course is (re)recorded last so a
    # bare "it" right after this turn still means the course, not one of
    # its instructors.
    _record_entity_list(
        memory, "faculty",
        [(source_url, name) for name, title, source_url in rows],
        "search_faculty_courses_taught",
    )
    _record_entity(
        memory, "course", normalized_code, label,
        "search_faculty_courses_taught",
    )

    display_rows = [(name, title) for name, title, source_url in rows]

    return (
        _format_faculty_list_context(
            "Faculty who have taught this course", label, display_rows
        ),
        None
    )


# Configurable topic-alias dictionary for person+topic course-taught
# queries (e.g. "Has X taught any AI courses?"). Fully deterministic -
# no embeddings anywhere in this feature. Each key is a natural-language
# topic phrase; each value is the list of phrases treated as equivalent
# when matching against course names. This is what lets a topic like
# "ai" match a course literally named "Artificial Intelligence" (no
# word overlap at all otherwise) without a semantic model - the
# tradeoff is that recall is capped by what's been added here. Keep
# alias entries as whole, specific phrases, never a single common short
# word - that's exactly the kind of entry that caused the department-
# name/conversation-detection false positives fixed earlier in this
# project.
_TOPIC_SYNONYMS = {
    "ai": ["ai", "artificial intelligence"],
    "machine learning": ["machine learning", "ml"],
    "database": ["database", "databases", "data management"],
}


def _topic_words(phrase):

    return frozenset(re.findall(r"[a-z]+", phrase.lower()))


# Built once from _TOPIC_SYNONYMS so the dictionary itself can stay in
# natural, human-editable phrasing (keys/aliases as plain strings) while
# lookups are order-insensitive (word-set based, not exact-string based).
_TOPIC_SYNONYMS_BY_WORDS = {
    _topic_words(key): aliases
    for key, aliases in _TOPIC_SYNONYMS.items()
}


def _expand_topic_aliases(topic):

    aliases = _TOPIC_SYNONYMS_BY_WORDS.get(_topic_words(topic))

    return aliases if aliases else [topic]


def _course_name_matches_alias(course_name, alias_phrase):

    course_words = _topic_words(course_name)
    alias_words = _topic_words(alias_phrase)

    # Containment, not equality - the reverse of the faculty/department
    # name-matching rule. There, an extra query word had to disqualify a
    # match (so "Faculty of Science" didn't swallow "Computer Science").
    # Here the course name is the longer, more descriptive side, so
    # requiring every alias word to appear somewhere in it (not the
    # other way around) is the correct direction, while still requiring
    # ALL of the alias's words - not just one - to rule out the same
    # single-short-word collision class fixed elsewhere in this project.
    return bool(alias_words) and alias_words.issubset(course_words)


# Deterministic "person + topic courses taught" intent detection. Two
# patterns cover the two orderings seen in practice ("Has X taught any
# Y courses?" and "What Y courses has X taught?"); anything else falls
# through unmatched rather than being guessed at. Both require the
# literal words "taught" and "course(s)" together, which is why this
# never collides with the plain "who has taught <code/name>" detector
# above (that one requires "who", which neither pattern here does) or
# the "who researches <topic>" detector (which never mentions "taught").
_PERSON_TAUGHT_TOPIC_PATTERN = re.compile(
    r"\bhas\s+.+?\s+taught\s+(?:any\s+)?(.+?)\s*courses?\b", re.IGNORECASE
)
_TOPIC_TAUGHT_PERSON_PATTERN = re.compile(
    r"\bwhat\s+(.+?)\s*courses?\s+has\s+.+?\s+taught\b", re.IGNORECASE
)


def _extract_person_topic_query(question):

    match = (
        _PERSON_TAUGHT_TOPIC_PATTERN.search(question)
        or _TOPIC_TAUGHT_PERSON_PATTERN.search(question)
    )

    if not match:
        return None

    topic = match.group(1).strip().rstrip("?.!, ")

    return topic or None


def search_faculty_courses_by_topic(question, memory=None):

    if not FACULTY_COURSES_TAUGHT_READY:
        return None

    topic = _extract_person_topic_query(question)

    if not topic:
        return None

    # Person resolution reuses the existing, already-proven name-matching
    # tiers rather than trying to isolate an exact "person phrase" from
    # the regex capture - that matching already handles credential
    # suffixes and surrounding text robustly and shouldn't be duplicated.
    person_row = search_faculty(question, memory)

    if not person_row:
        return (
            "I couldn't identify a specific faculty member in that "
            "question.",
            None
        )

    person_name = person_row[0]
    person_source_url = person_row[9]

    conn = sqlite3.connect("data/faculty.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT course_code FROM faculty_courses_taught "
        "WHERE faculty_source_url = ?",
        (person_source_url,)
    )

    codes = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not codes:
        return (
            f"No course-taught information is available for "
            f"{person_name}.",
            None
        )

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(codes))

    cursor.execute(
        f"SELECT course_code, course_name FROM courses "
        f"WHERE course_code IN ({placeholders})",
        codes
    )

    course_rows = cursor.fetchall()

    conn.close()

    aliases = _expand_topic_aliases(topic)
    topic_words = _topic_words(topic)

    tier1 = []
    tier2 = []
    seen_codes = set()

    for code, course_name in course_rows:

        if not course_name or code in seen_codes:
            continue

        # Tier 1: the literal topic phrase, as typed, matches.
        if _course_name_matches_alias(course_name, topic):
            tier1.append((code, course_name, None))
            seen_codes.add(code)
            continue

        # Tier 2: only a dictionary alias (not the literal phrase)
        # matches.
        for alias in aliases:

            if _topic_words(alias) == topic_words:
                continue

            if _course_name_matches_alias(course_name, alias):
                tier2.append((code, course_name, alias))
                seen_codes.add(code)
                break

    # Deterministic ranking: literal matches before synonym matches, no
    # scores involved - just course_code as a fixed, content-independent
    # tiebreaker within each tier.
    tier1.sort(key=lambda row: row[0])
    tier2.sort(key=lambda row: row[0])

    matches = tier1 + tier2

    if not matches:
        return (
            f'{person_name} has taught courses, but none found related '
            f'to "{topic}".',
            None
        )

    # person_row was already recorded as a faculty entity inside
    # search_faculty() above; only the returned courses are new here.
    _record_entity_list(
        memory, "course",
        [(code, course_name) for code, course_name, alias in matches],
        "search_faculty_courses_by_topic",
    )

    lines = []

    for code, course_name, alias in matches:

        if alias:
            lines.append(f'- {code} - {course_name} (matched via "{alias}")')
        else:
            lines.append(f"- {code} - {course_name}")

    course_lines = "\n".join(lines)

    context = f"""
Faculty member: {person_name}

Courses taught related to "{topic}":
{course_lines}
"""

    return context, None


# Generic normalization rules for program-name substring matching.
# "Honours"/"Program"/"Degree" are pure qualifiers and are stripped
# entirely. Degree-type phrases ("Bachelor of", "Master of", ...) are
# collapsed to a short canonical token rather than deleted outright -
# deleting them entirely would make "Bachelor of X" and "Master of X"
# normalize to the same text and collide with each other, which is wrong
# (a "Bachelor of" query should never resolve to a "Master of" program).
_PROGRAM_NORMALIZE_RULES = [
    (r"\bhonours\b", " "),
    (r"\bhonors\b", " "),
    (r"\bbachelor of\b", " bach "),
    (r"\bmaster of\b", " mast "),
    (r"\bdoctor of\b", " doc "),
    (r"\bdiploma in\b", " dip "),
    # Abbreviated bachelor's-degree prefixes (Sprint 11B) - undergraduate
    # program titles almost always use these short forms ("Honours BSc
    # Computer Science") rather than spelling out "Bachelor of Science
    # in Computer Science" the way graduate titles spell out "Master
    # of...". Collapsed to the same "bach" token as "bachelor of" above,
    # since they're the same credential level - confirmed against the
    # real prefixes present in the discovered undergraduate catalog
    # (BA, BBA, BKin, BMus, BSc).
    (r"\bb(?:a|ba|kin|mus|sc)\b", " bach "),
    (r"\bprogram\b", " "),
    (r"\bdegree\b", " "),
]

# For acronym generation we keep degree-type words (Bachelor/Master/Doctor/
# Diploma all contribute a letter, e.g. the "B" in "BBA") but still strip
# pure qualifiers that aren't part of the conventional abbreviation.
_ACRONYM_QUALIFIER_PHRASES = [
    "honours",
    "honors",
    "program",
    "degree",
]

_ACRONYM_SKIP_WORDS = {"of", "in", "and", "the", "for", "with", "a", "an"}


def _strip_filler(text):

    text = text.lower()

    for pattern, replacement in _PROGRAM_NORMALIZE_RULES:
        text = re.sub(pattern, replacement, text)

    return re.sub(r"\s+", " ", text).strip()


# Sprint 11B: _strip_filler() collapses degree-type phrases to a short
# canonical token ("bach"/"mast"/...) rather than deleting them, on
# purpose, so "Bachelor of X" and "Master of X" can never collide with
# each other. That's exactly right for graduate titles, which always
# spell the degree type out in full ("Master of Applied Computing") -
# but undergraduate titles are consistently degree-prefixed too
# ("Honours BSc Computer Science"), and users overwhelmingly just name
# the bare subject ("Computer Science") without mentioning any degree
# type at all. This goes one step further, stripping the collapsed
# degree token itself, for a bare-subject fallback tried only after the
# degree-aware Tier 2 match fails (search_program()).
_DEGREE_TOKEN_PATTERN = re.compile(r"\b(?:bach|mast|doc|dip)\b")


def _strip_to_subject(text):

    return re.sub(r"\s+", " ", _DEGREE_TOKEN_PATTERN.sub(" ", _strip_filler(text))).strip()


# Bare single-word subjects ("Philosophy", "Music", "History", "English",
# "Psychology", "Biology" - all real undergraduate program subjects,
# confirmed live) collide with ordinary English words the exact same way
# single-word DEPARTMENT names do (Sprint 5C's "I love music"/"do you
# speak English" guard) - confirmed live during Sprint 11B verification:
# without this guard, "What is the philosophy behind this decision?" and
# "I love music." both incorrectly resolved to a program. Deliberately
# does NOT include "department" as a qualifying signal, unlike the
# department guard - a question naming both a subject and the word
# "department" ("Who coordinates the History department?") is asking
# about the DEPARTMENT, not the program, and must be left for
# search_department() to handle instead; including "department" here
# would silently re-break that routing.
_PROGRAM_SUBJECT_SIGNAL_PATTERN = re.compile(
    r"\b(?:program|major|minor|degree|concentration|option|certificate|"
    r"stud(?:y|ying)|undergraduate|at\s+laurier|at\s+wlu|"
    r"at\s+wilfrid\s+laurier)\b",
    re.IGNORECASE
)


def _subject_match_is_safe(subject, question_lower):

    if " " in subject:
        return True

    return bool(_PROGRAM_SUBJECT_SIGNAL_PATTERN.search(question_lower))


def _generate_acronym(program_name):

    text = program_name.lower()

    for phrase in _ACRONYM_QUALIFIER_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)

    words = re.findall(r"[a-zA-Z]+", text)

    letters = [w[0] for w in words if w not in _ACRONYM_SKIP_WORDS]

    return "".join(letters).upper()


def search_program(question, memory=None):

    conn = sqlite3.connect(
        "data/programs.db"
    )

    # Sprint 11B: `program_type` and `description` are appended after
    # the conditionally-present `level` column - sqlite3.Row keeps every
    # existing positional access (row[0], row[3], ...) working unchanged
    # while giving safe, unambiguous name-based access to the new
    # columns regardless of whether `level` shifted their position
    # (same pattern established in Sprint 10D/10E for courses/
    # departments).
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    level_column = ", level" if PROGRAMS_HAVE_LEVEL else ""

    cursor.execute(f"""
    SELECT
        program_name,
        admission_requirements,
        program_requirements,
        source_url
        {level_column},
        program_type,
        description
    FROM programs
    """)

    rows = cursor.fetchall()

    conn.close()

    question_lower = question.lower()

    # Tier 1: exact stored-name substring match (highest priority, unchanged).
    for row in rows:

        if row[0].lower() in question_lower:

            if memory is not None:
                memory["last_program"] = row[0]
                _record_entity(memory, "program", row[0], row[0], "search_program")

            return row

    # Tier 2: fallback - normalized-phrase match or generic acronym match,
    # so users don't need to type the exact official title. Nothing here is
    # specific to any one program; both are derived generically from
    # whatever program_name is stored.
    normalized_question = _strip_filler(question_lower)

    for row in rows:

        program_name = row[0]

        normalized_name = _strip_filler(program_name.lower())

        if len(normalized_name) >= 3 and normalized_name in normalized_question:

            if memory is not None:
                memory["last_program"] = row[0]
                _record_entity(memory, "program", row[0], row[0], "search_program")

            return row

        acronym = _generate_acronym(program_name)

        # Require at least 3 letters and an exact-case match against the
        # ORIGINAL question (not lowercased). Real acronym usage is almost
        # always typed uppercase ("BBA", "MBA"); matching case-insensitively
        # causes false positives against ordinary lowercase words that
        # happen to coincide with a short acronym (e.g. "map" the verb vs.
        # "MAP" for Master of Applied Politics, or "me" vs "ME").
        if len(acronym) >= 3 and re.search(
            rf"\b{re.escape(acronym)}\b", question
        ):

            if memory is not None:
                memory["last_program"] = row[0]
                _record_entity(memory, "program", row[0], row[0], "search_program")

            return row

    # Tier 2b: bare-subject fallback (Sprint 11B) - see _strip_to_subject()
    # docstring. Tried only after the degree-aware Tier 2 match above
    # fails. A bare subject name is often shared by several program-type
    # variants of the same subject (major, minor, concentration,
    # combined...), so this is checked in two passes: "major"-type rows
    # first, so a bare subject name resolves to the plain major by
    # default (per Sprint 11A's investigation recommendation) rather
    # than an arbitrarily-ordered concentration/option/minor variant;
    # any other type only as a second-pass fallback.
    for preferred_types in (("major",), None):

        for row in rows:

            program_type = row["program_type"] if "program_type" in row.keys() else None

            if preferred_types and program_type not in preferred_types:
                continue

            subject = _strip_to_subject(row[0].lower())

            if (
                len(subject) >= 3
                and subject in normalized_question
                and _subject_match_is_safe(subject, question_lower)
            ):

                if memory is not None:
                    memory["last_program"] = row[0]
                    _record_entity(memory, "program", row[0], row[0], "search_program")

                return row

    return None


# Deterministic "list the undergraduate catalog" intent (Sprint 11B) -
# deliberately narrow (requires "undergraduate" explicitly, plus
# "program(s)" and one of a small set of listing-style words) so it
# never collides with a specific single-program question like "Tell me
# about the undergraduate Computer Science program", which doesn't ask
# what's available/offered/list-worthy, just names one program directly.
def _has_undergraduate_program_list_intent(question_lower):

    return bool(
        "undergraduate" in question_lower
        and re.search(r"\bprograms?\b", question_lower)
        and re.search(r"\b(?:available|offered|exist|list|what)\b", question_lower)
    )


def search_undergraduate_program_list(question, memory=None):

    if not _has_undergraduate_program_list_intent(question.lower()):
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM programs "
        "WHERE level = 'undergraduate' ORDER BY program_name"
    )

    names = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not names:
        return None

    total = len(names)
    displayed = names[:30]

    lines = "\n".join(f"- {name}" for name in displayed)

    truncation_note = (
        f"\n(Showing {len(displayed)} of {total} undergraduate programs. "
        f"Ask about a specific program, major, or minor by name for more detail.)"
        if total > len(displayed) else ""
    )

    return (
        f"Undergraduate programs at Wilfrid Laurier University "
        f"include:\n{lines}\n{truncation_note}",
        None
    )


# Signals the user wants the program's *coordinator* specifically, not
# just general program information - e.g. "Who is the program coordinator
# for the Master of Applied Computing?", "Who coordinates the MBA?". A
# prefix match covers every inflection (coordinator/coordinators/
# coordinates/coordinate/coordinating) without enumerating them.
_COORDINATOR_INTENT_PATTERN = re.compile(r"\bcoordinat\w*\b")


def _has_coordinator_intent(question_lower):

    return bool(_COORDINATOR_INTENT_PATTERN.search(question_lower))


# programs.db has no department_name/coordinator field of its own, but
# both programs.db and departments.db source URLs come from the same
# academic-calendar.wlu.ca site and carry the same "d=<id>" department-id
# query parameter - so a program can be joined to its owning department
# without any new scraping or schema change, just by matching that id
# between the two already-scraped source_url values.
def _extract_department_id(url):

    if not url:
        return None

    match = re.search(r"[?&]d=(\d+)", url)

    return match.group(1) if match else None


def _get_department_coordinator(program_source_url):

    department_id = _extract_department_id(program_source_url)

    if not department_id:
        return None

    conn = sqlite3.connect(
        "data/departments.db"
    )

    cursor = conn.cursor()

    cursor.execute("SELECT coordinator, source_url FROM departments")

    rows = cursor.fetchall()

    conn.close()

    for coordinator, dept_source_url in rows:

        if _extract_department_id(dept_source_url) == department_id:

            if coordinator and coordinator.strip():
                return coordinator.strip()

            return None

    return None


# Single-word department names collide with ordinary English words used
# completely outside any WLU context - "English", "History", "Music",
# "Philosophy", "Psychology", "Biology", "Business", "Economics",
# "Education", "Sociology" are all real departments.db entries that are
# also common words ("Do you speak English?", "I love music."). A bare
# word-boundary match on the name alone isn't enough evidence the user
# means the academic department, so single-word names additionally
# require one of these academic-context signals to be present. Multi-
# word department names ("Physics and Computer Science") are already
# specific enough that a whole-phrase match alone is safe - this gate
# only applies to the single-word case.
# "coordinat*" (Sprint 10E) is included here too: "who is the
# coordinator of Biology?" has no other academic-signal word at all, but
# asking about an academic coordinator is itself never a coincidental,
# non-WLU usage the way "history"/"music"/"english" commonly are.
_DEPARTMENT_ACADEMIC_SIGNAL_PATTERN = re.compile(
    r"\b(?:department|faculty\s+of|program|major|minor|degree|coordinat\w*|"
    r"at\s+laurier|at\s+wlu|at\s+wilfrid\s+laurier)\b",
    re.IGNORECASE
)


def _department_name_matches(department_name, question_lower):

    # Whole-word/whole-phrase match, not substring containment - this is
    # what stops a name like "Art" from matching inside an unrelated word,
    # on top of the academic-signal gate below.
    if not re.search(
        rf"\b{re.escape(department_name.lower())}\b", question_lower
    ):
        return False

    if " " in department_name.strip():
        return True

    return bool(_DEPARTMENT_ACADEMIC_SIGNAL_PATTERN.search(question_lower))


def search_department(question, memory=None):

    conn = sqlite3.connect(
        "data/departments.db"
    )

    # Sprint 10E: `coordinator` is appended after the conditionally-
    # present `level` column (same shape as courses.db in Sprint 10D) -
    # sqlite3.Row keeps every existing positional access (result[0],
    # result[3], ...) working unchanged while giving safe, unambiguous
    # name-based access to `coordinator` regardless of whether `level`
    # shifted its position.
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    level_column = ", level" if DEPARTMENTS_HAVE_LEVEL else ""

    cursor.execute(f"""
    SELECT
        department_name,
        programs,
        description,
        source_url
        {level_column},
        coordinator
    FROM departments
    """)

    rows = cursor.fetchall()

    conn.close()

    question_lower = question.lower()

    for row in rows:

        if _department_name_matches(row[0], question_lower):

            if memory is not None:
                memory["last_department"] = row[0]
                _record_entity(memory, "department", row[0], row[0], "search_department")

            return row

    return None


# Common title words - stripped for name matching, and required (alongside
# a capitalized surname) for the surname-only fallback tier below.
_PERSON_TITLE_WORDS = ["dr.", "dr", "professor", "prof.", "prof"]


def _strip_person_titles(text):

    text = text.lower()

    # Matched separately from _PERSON_TITLE_WORDS: a trailing "\b" after an
    # escaped period never matches (a period followed by a space has no
    # word boundary on either side), which left a stray "." behind and
    # broke the substring match this function exists to enable. The "X."
    # alternative has no trailing boundary requirement - the literal
    # period is itself an unambiguous delimiter.
    text = re.sub(r"\bdr\.|\bdr\b", " ", text)
    text = re.sub(r"\bprof\.|\bprof\b", " ", text)
    text = re.sub(r"\bprofessor\b", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def _strip_credentials(name):

    # Stored names sometimes carry a trailing academic/professional
    # credential after a comma (e.g. "Matthew Smith, PhD", "Jane Doe, MD",
    # "John Smith, P.Eng.") - stripping at the first comma is generic to
    # any credential without needing to enumerate them, and leaves names
    # with no comma (the common case) untouched.
    return name.split(",")[0].strip()


# Signals a "list of people" intent rather than a single-department-info
# or single-person intent - e.g. "Who works in Marketing?", "Who are the
# Accounting faculty?", "List Computer Science faculty." This is checked
# BEFORE any department-name matching happens; the name match alone is
# what actually decides whether a real result comes back, so a broad
# trigger here (e.g. "who is") is safe; it just costs one extra check on
# queries that don't turn out to reference any real department.
_DEPARTMENT_LIST_TRIGGER_PATTERNS = [
    r"\bwho\s+(?:works|work|teaches|teach|is|are)\b",
    r"\blist\b",
]

# Pure function/filler words stripped from the question before comparing
# whatever's left against a stored department name. This is what lets
# "List Computer Science faculty" resolve against the stored department
# "Computer Science and Physics" even though neither string literally
# contains the other in full.
_DEPARTMENT_LIST_FILLER_WORDS = {
    "who", "works", "work", "teaches", "teach", "is", "are", "the",
    "in", "at", "of", "for", "list", "faculty", "professors",
    "professor", "staff", "members", "member", "department",
    "departments", "please", "tell", "me", "about", "what",
}


def _has_department_list_intent(question_lower):

    return any(
        re.search(pattern, question_lower)
        for pattern in _DEPARTMENT_LIST_TRIGGER_PATTERNS
    )


def _department_list_residual(question_lower):

    words = re.findall(r"[a-z]+", question_lower)

    remaining = [w for w in words if w not in _DEPARTMENT_LIST_FILLER_WORDS]

    return " ".join(remaining)


# Cross-listed faculty store multiple department affiliations in one
# " | "-joined field (see get_faculty_links.py) - splitting on that same
# delimiter recovers the individual real department names to match
# against, rather than requiring the full joined string as one unit.
def _department_name_segments(cursor):

    cursor.execute("SELECT DISTINCT department_name FROM faculty")

    segments = set()

    for (value,) in cursor.fetchall():
        for part in value.split(" | "):
            part = part.strip()
            if part:
                segments.add(part)

    return sorted(segments, key=len, reverse=True)


def search_faculty_by_department(question, memory=None):

    if not FACULTY_DB_READY:
        return None

    question_lower = question.lower()

    if not _has_department_list_intent(question_lower):
        return None

    conn = sqlite3.connect(
        "data/faculty.db"
    )

    cursor = conn.cursor()

    segments = _department_name_segments(cursor)

    residual = _department_list_residual(question_lower)

    matched_segment = None

    for segment in segments:

        segment_lower = segment.lower()

        if len(segment_lower) < 3:
            continue

        if segment_lower in question_lower:
            matched_segment = segment
            break

        if len(residual) >= 3 and (
            residual in segment_lower or segment_lower in residual
        ):
            matched_segment = segment
            break

    if not matched_segment:
        conn.close()
        return None

    cursor.execute(
        "SELECT name, title, source_url FROM faculty "
        "WHERE department_name LIKE ? ORDER BY name",
        (f"%{matched_segment}%",)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return None

    _record_entity_list(
        memory, "faculty",
        [(source_url, name) for name, title, source_url in rows],
        "search_faculty_by_department",
    )
    _record_entity(
        memory, "department", matched_segment, matched_segment,
        "search_faculty_by_department",
    )

    display_rows = [(name, title) for name, title, source_url in rows]

    return matched_segment, display_rows


# Faculty-level names ("Faculty of Science", "Faculty of Arts",
# "Lazaridis School of Business and Economics") are a small, fixed set of
# known institutional names, stored in the faculty_name column - a
# different column from department_name. Matching them the same way
# department names are matched (substring/residual containment) is what
# let a bare residual word like "science" collide with an unrelated
# department ("...Decision Sciences"), so this is a separate, stricter
# mechanism: every significant word of a stored faculty name must appear
# as a whole word in the question. Word-level equality (not substring
# containment) means "science" and "sciences" are simply different words
# and can't collide, and requiring ALL of a name's words - not just one -
# rules out a short name matching whenever a longer, related name would
# also fit.
_FACULTY_LEVEL_FILLER_WORDS = _DEPARTMENT_LIST_FILLER_WORDS | {"school", "and"}


def _significant_words(text, filler_words):

    words = re.findall(r"[a-z]+", text.lower())

    return {w for w in words if w not in filler_words}


def _faculty_name_segments(cursor):

    cursor.execute("SELECT DISTINCT faculty_name FROM faculty")

    segments = set()

    for (value,) in cursor.fetchall():
        for part in value.split(" | "):
            part = part.strip()
            if part:
                segments.add(part)

    return segments


def search_faculty_by_faculty_name(question, memory=None):

    if not FACULTY_DB_READY:
        return None

    question_lower = question.lower()

    if not _has_department_list_intent(question_lower):
        return None

    conn = sqlite3.connect(
        "data/faculty.db"
    )

    cursor = conn.cursor()

    segments = _faculty_name_segments(cursor)

    question_words = _significant_words(question, _FACULTY_LEVEL_FILLER_WORDS)

    # Exact set equality, not subset containment: a subset check would let
    # "Faculty of Science" (word set {"science"}) match "Computer Science"
    # too, since {"science"} is a subset of {"computer", "science"} - which
    # would wrongly steal a department-level query. Requiring the two word
    # sets to match exactly means an extra word like "computer" correctly
    # rules the faculty-level name out, leaving it to department matching.
    candidates = [
        segment
        for segment in segments
        if _significant_words(segment, _FACULTY_LEVEL_FILLER_WORDS) == question_words
        and question_words
    ]

    if not candidates:
        conn.close()
        return None

    # Any tie is broken alphabetically - never by set/dict iteration
    # order, which Python randomizes per process - so the result is
    # stable across runs.
    candidates.sort()

    matched_segment = candidates[0]

    cursor.execute(
        "SELECT name, title, source_url FROM faculty "
        "WHERE faculty_name LIKE ? ORDER BY name",
        (f"%{matched_segment}%",)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return None

    _record_entity_list(
        memory, "faculty",
        [(source_url, name) for name, title, source_url in rows],
        "search_faculty_by_faculty_name",
    )
    _record_entity(
        memory, "faculty_institution", matched_segment, matched_segment,
        "search_faculty_by_faculty_name",
    )

    display_rows = [(name, title) for name, title, source_url in rows]

    return matched_segment, display_rows


def search_faculty(question, memory=None):

    if not FACULTY_DB_READY:
        return None

    conn = sqlite3.connect(
        "data/faculty.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        name,
        title,
        faculty_name,
        department_name,
        email,
        phone,
        office,
        research_interests,
        biography,
        source_url
    FROM faculty
    """)

    rows = cursor.fetchall()

    conn.close()

    question_lower = question.lower()

    # Tier 1: title-stripped full-name substring match (highest priority).
    normalized_question = _strip_person_titles(question_lower)

    for row in rows:

        normalized_name = _strip_person_titles(_strip_credentials(row[0]).lower())

        if len(normalized_name) >= 4 and normalized_name in normalized_question:

            if memory is not None:
                memory["last_faculty"] = row[0]
                _record_entity(memory, "faculty", row[9], row[0], "search_faculty")
                if row[3]:
                    _record_entity(memory, "department", row[3], row[3], "search_faculty")

            return row

    # Tier 2: fallback - first name AND last name both present, capitalized,
    # as whole words in the original question (e.g. "Tell me about Louise
    # Dawe" - no title, and the stored "Dr. Louise N. Dawe" won't substring
    # -match because of the middle initial). No title word required here,
    # but requiring two independent capitalized-word matches (rather than
    # one) keeps false-positive risk low without needing a title as a
    # safety net.
    for row in rows:

        name_parts = _strip_person_titles(_strip_credentials(row[0]).lower()).split()

        if len(name_parts) < 2:
            continue

        first_name, last_name = name_parts[0], name_parts[-1]

        if len(first_name) < 3 or len(last_name) < 3:
            continue

        first_pattern = rf"\b{re.escape(first_name.capitalize())}\b"
        last_pattern = rf"\b{re.escape(last_name.capitalize())}\b"

        if re.search(first_pattern, question) and re.search(last_pattern, question):

            if memory is not None:
                memory["last_faculty"] = row[0]
                _record_entity(memory, "faculty", row[9], row[0], "search_faculty")
                if row[3]:
                    _record_entity(memory, "department", row[3], row[3], "search_faculty")

            return row

    # Tier 3: fallback - surname only (e.g. "Who is Professor Ghose?").
    # Requires BOTH a title word present in the question AND the surname
    # appearing capitalized as a whole word in the ORIGINAL (non-lowercased)
    # question. Plain case-insensitive surname matching would collide with
    # ordinary English words that are also real surnames here (e.g. "Gates",
    # "Long") - the same lesson learned from acronym matching for programs.
    has_title_word = any(
        re.search(rf"\b{re.escape(word)}\b", question_lower)
        for word in _PERSON_TITLE_WORDS
    )

    if has_title_word:

        for row in rows:

            name_parts = _strip_credentials(row[0]).strip().split()

            if not name_parts:
                continue

            surname = name_parts[-1]

            if len(surname) >= 3 and re.search(
                rf"\b{re.escape(surname)}\b", question
            ):

                if memory is not None:
                    memory["last_faculty"] = row[0]
                    _record_entity(memory, "faculty", row[9], row[0], "search_faculty")
                    if row[3]:
                        _record_entity(memory, "department", row[3], row[3], "search_faculty")

                return row

    return None


# Deterministic research-topic intent detection - same style as the
# department-list/coordinator detectors above, but this one also extracts
# the topic text itself (the capture group), since there's no fixed list
# of known values to match against the way there is for departments or
# faculties. No LLM classification is used anywhere in this function.
_RESEARCH_INTENT_PATTERNS = [
    re.compile(
        r"who\s+(?:researches|studies|works on|specializes in|"
        r"is interested in|does research (?:on|in))\s+(.+)",
        re.IGNORECASE
    ),
    re.compile(r"research(?:ers?)?\s+(?:on|in|about)\s+(.+)", re.IGNORECASE),
    re.compile(
        r"i want to (?:study|research|learn about)\s+(.+)", re.IGNORECASE
    ),
    re.compile(r"(?:expertise|specialization)\s+(?:in|on)\s+(.+)", re.IGNORECASE),
]

# Trailing institution phrases stripped from the captured topic text so
# "who researches AI at Laurier?" resolves to just "AI".
_RESEARCH_TOPIC_TRAILING_PHRASES = (
    "at wilfrid laurier university", "at laurier", "at wlu", "here",
)

# Chosen from real distance data gathered against the faculty-research
# collection across several representative topics (quantum computing,
# machine learning, consumer behavior, artificial intelligence, AI) -
# genuine topical matches consistently land below ~1.0, while the tail
# beyond that mixes in progressively weaker/unrelated matches. Kept as a
# named constant since it's a calibrated value, not an arbitrary one.
_RESEARCH_TOPIC_DISTANCE_THRESHOLD = 1.0


def _extract_research_topic(question):

    for pattern in _RESEARCH_INTENT_PATTERNS:

        match = pattern.search(question)

        if not match:
            continue

        topic = match.group(1).strip().rstrip("?.!, ")
        topic_lower = topic.lower()

        for phrase in _RESEARCH_TOPIC_TRAILING_PHRASES:
            if topic_lower.endswith(phrase):
                topic = topic[:len(topic) - len(phrase)].strip()
                break

        topic = topic.rstrip("?.!, ")

        if len(topic) >= 2:
            return topic

    return None


def search_faculty_by_research_topic(question, memory=None):

    if not FACULTY_RESEARCH_READY:
        return None

    topic = _extract_research_topic(question)

    if not topic:
        return None

    # Phrased to match the corpus's own first-person research-statement
    # style ("I am interested in...", "My research focuses on...") rather
    # than embedding the bare noun phrase - verified against real data to
    # noticeably tighten distances and improve ranking quality.
    query_text = f"research interests in {topic}"

    embedding = model.encode(query_text).tolist()

    results = faculty_research_collection.query(
        query_embeddings=[embedding],
        n_results=10,
        include=["metadatas", "distances"]
    )

    # Structured retrieval results first (source_urls + distances) - the
    # similarity threshold is applied here, before any SQLite access.
    # source_url is the only persistent key read from Chroma metadata -
    # unlike faculty.id, it stays valid across a faculty.db reload, since
    # load_faculty.py reassigns ids (DELETE + re-insert) but never changes
    # a profile's own URL.
    candidate_urls = [
        meta["source_url"]
        for meta, distance in zip(
            results["metadatas"][0], results["distances"][0]
        )
        if distance <= _RESEARCH_TOPIC_DISTANCE_THRESHOLD
    ]

    if not candidate_urls:
        return None

    # Re-fetch the authoritative rows from SQLite - Chroma only ever
    # decided *which* profiles are relevant, never what to display for
    # them.
    conn = sqlite3.connect(
        "data/faculty.db"
    )

    cursor = conn.cursor()

    placeholders = ",".join("?" * len(candidate_urls))

    cursor.execute(
        f"SELECT source_url, name, title FROM faculty "
        f"WHERE source_url IN ({placeholders})",
        candidate_urls
    )

    fetched = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    conn.close()

    # Preserve Chroma's relevance ordering rather than SQL's row order.
    rows = [(u, fetched[u][0], fetched[u][1]) for u in candidate_urls if u in fetched]

    if not rows:
        return None

    _record_entity_list(
        memory, "faculty",
        [(url, name) for url, name, title in rows],
        "search_faculty_by_research_topic",
    )

    display_rows = [(name, title) for url, name, title in rows]

    return topic, display_rows


def search_vector(question):

    embedding = model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    return results


def _level_line(result, index):

    # '' if the level column wasn't queried (old, pre-undergraduate schema)
    if len(result) <= index or not result[index]:
        return ""

    return f"Level: {result[index].capitalize()}\n"


# _level_line(result, index) is still shared with search_program(),
# which returns a plain tuple with nothing appended after its own
# conditional level column - its length-based "wasn't queried" check is
# correct and unaffected there. Both COURSE (Sprint 10D) and DEPARTMENT
# (Sprint 10E) rows now have more columns appended after where `level`
# conditionally sits, so that length check would never trigger for them
# even when level is genuinely absent, and reading a fixed index would
# silently read the wrong column instead. sqlite3.Row's name-based
# lookup sidesteps the whole positional-shift problem for both.
def _course_level_line(result):

    if "level" not in result.keys() or not result["level"]:
        return ""

    return f"Level: {result['level'].capitalize()}\n"


def _department_level_line(result):

    if "level" not in result.keys() or not result["level"]:
        return ""

    return f"Level: {result['level'].capitalize()}\n"


def _program_level_line(result):

    if "level" not in result.keys() or not result["level"]:
        return ""

    return f"Level: {result['level'].capitalize()}\n"


def _program_type_line(result):

    if "program_type" not in result.keys() or not result["program_type"]:
        return ""

    return f"Program Type: {result['program_type'].capitalize()}\n"


# Sprint 10D: surfaces courses.db metadata that was already captured by
# the scraper but never shown anywhere - purely additive to
# search_course()'s existing context, no new capability or routing.
# Fixed, consistent order regardless of which fields happen to be
# present; a field with no data simply produces no line at all, so a
# course with none of these five populated renders identically to
# before this sprint.
_COURSE_METADATA_FIELDS = [
    ("Prerequisites", "prerequisites_text"),
    ("Corequisites", "corequisites_text"),
    ("Exclusions", "exclusions_text"),
    ("Location", "location_text"),
    ("Notes", "notes_text"),
]


def _course_metadata_section(result):

    lines = [
        f"{label}: {result[column].strip()}"
        for label, column in _COURSE_METADATA_FIELDS
        if result[column] and result[column].strip()
    ]

    if not lines:
        return ""

    return "\n" + "\n".join(lines) + "\n"


def _format_faculty_list_context(label, name, rows):

    total = len(rows)
    displayed = rows[:25]

    faculty_lines = "\n".join(
        f"- {faculty_name} ({title})" if title else f"- {faculty_name}"
        for faculty_name, title in displayed
    )

    truncation_note = (
        f"\n(Showing {len(displayed)} of {total} faculty members.)"
        if total > len(displayed) else ""
    )

    return f"""
{label}: {name}

Faculty members:
{faculty_lines}
{truncation_note}
"""


def structured_search(question, memory=None):

    # FOLLOWUP MEMORY

    question_lower = question.lower()

    if memory is not None and normalize_followup_text(question) in FOLLOWUP_PHRASES:

        if memory.get("last_course"):
            question = memory["last_course"]

        elif memory.get("last_program"):
            question = memory["last_program"]

        elif memory.get("last_department"):
            question = memory["last_department"]

        elif memory.get("last_faculty"):
            question = memory["last_faculty"]

    # FACULTY COURSES TAUGHT
    # Must run before the plain COURSE lookup below: "Who has taught
    # CP104?" contains a bare course code that search_course() would
    # otherwise match first, returning the course description instead of
    # answering who taught it. Bare "What is CP104?" has no "who has
    # taught" trigger, so it's unaffected and still reaches search_course
    # exactly as before.

    taught_result = search_faculty_courses_taught(question, memory)

    if taught_result:
        return taught_result

    # FACULTY COURSES TAUGHT - PERSON + TOPIC
    # Must also run before the plain single-person FACULTY lookup further
    # below: "Has Kaiyu Li taught database courses?" contains a full
    # name that search_faculty()'s own tiered matching would otherwise
    # match directly, returning a generic profile instead of answering
    # the actual question.

    person_topic_result = search_faculty_courses_by_topic(question, memory)

    if person_topic_result:
        return person_topic_result

    # COURSE PREREQUISITES
    # Must run before the plain COURSE lookup below: "What are the
    # prerequisites for CP600?" and "Does CP312 require CP220?" both
    # contain a bare course code that search_course() would otherwise
    # match first, returning the course description instead of
    # answering the actual question.

    prerequisite_result = search_course_prerequisites(question, memory)

    if prerequisite_result:
        return prerequisite_result

    # PROGRAM COURSE REQUIREMENTS (graduate only)
    # Must also run before both the plain COURSE and PROGRAM lookups
    # below: "Does the Master of Applied Computing require CP600?" and
    # "Which required courses are listed for the Master of Applied
    # Computing?" both contain a bare course code and/or an exact
    # program name that those plain lookups would otherwise match first.

    program_requirement_result = search_program_course_requirements(
        question, memory
    )

    if program_requirement_result:
        return program_requirement_result

    # COURSE

    result = search_course(question, memory)

    if result:

        context = f"""
Course Code: {result[0]}
Course Name: {result[1]}
Credits: {result[2]}
Department: {result[4]}
{_course_level_line(result)}
Description:
{result[3]}
{_course_metadata_section(result)}"""

        return context, result[5]

    # UNDERGRADUATE PROGRAM LIST (Sprint 11B)
    # Checked before the single-program lookup below: "What undergraduate
    # programs are available?" doesn't name any specific program, so it
    # would never match search_program() anyway, but is checked first
    # for the same reason department-list intent is checked before a
    # single-department lookup elsewhere in this cascade - a listing
    # request and a single-item lookup are different capabilities
    # answering different questions.

    program_list_result = search_undergraduate_program_list(question, memory)

    if program_list_result:
        return program_list_result

    # PROGRAM

    result = search_program(question, memory)

    if result:

        context = f"Program: {result[0]}\n{_program_level_line(result)}{_program_type_line(result)}"

        # Sprint 11B: Description/Admission Requirements/Program
        # Requirements are now each shown only when actually populated
        # (same "only include sections with data" principle established
        # for course metadata in Sprint 10D) - undergraduate programs
        # deliberately have no program_requirements yet (course-
        # requirement extraction is out of scope this sprint) and most
        # have no admission_requirements at all (confirmed live: no
        # undergraduate program page publishes per-program admission
        # content), so showing empty labeled sections for every one of
        # the 399 new rows would be misleading clutter. Every existing
        # graduate program already has non-empty values for both
        # fields, so this changes nothing for them.
        description = result["description"] if "description" in result.keys() else None

        if description and description.strip():
            context += f"\nDescription:\n{description.strip()}\n"

        admission = result[1]

        if admission and admission.strip():
            context += f"\nAdmission Requirements:\n{admission.strip()}\n"

        requirements = result[2]

        if requirements and requirements.strip():
            context += f"\nProgram Requirements:\n{requirements.strip()}\n"

        # Coordinator info is only added when specifically asked for, so
        # every other program query keeps producing exactly the context
        # above, unchanged.
        if _has_coordinator_intent(question_lower):

            coordinator = _get_department_coordinator(result[3])

            context += (
                f"\nProgram Coordinator:\n{coordinator}\n"
                if coordinator else
                "\nProgram Coordinator: Coordinator information is not available.\n"
            )

        return context, result[3]

    # FACULTY-LEVEL LIST (e.g. "Faculty of Science", "Faculty of Arts")
    # Tried before the department-level list check: these are a small,
    # fixed set of known institutional names matched deterministically by
    # exact word-set comparison against faculty_name, not the
    # substring/residual matching department names use. Skipped when
    # coordinator intent is present (Sprint 10E): "who is the
    # coordinator of X" incidentally matches this trigger's broad
    # "who is" phrasing (_has_department_list_intent), but is asking
    # about one specific role, not a list of people - left for the
    # DEPARTMENT section below, which handles coordinator lookup
    # directly.

    faculty_level_result = (
        None if _has_coordinator_intent(question_lower)
        else search_faculty_by_faculty_name(question, memory)
    )

    if faculty_level_result:

        matched_faculty, faculty_rows = faculty_level_result

        return (
            _format_faculty_list_context("Faculty", matched_faculty, faculty_rows),
            None
        )

    # DEPARTMENT - FACULTY LIST
    # Must run before the single-department lookup below: both can match
    # on the same department name, but a "who works in X" / "list X
    # faculty" query wants the list of people, not the department's own
    # generic description. Also skipped when coordinator intent is
    # present, for the same reason as the faculty-level list above.

    dept_list_result = (
        None if _has_coordinator_intent(question_lower)
        else search_faculty_by_department(question, memory)
    )

    if dept_list_result:

        matched_department, faculty_rows = dept_list_result

        return (
            _format_faculty_list_context("Department", matched_department, faculty_rows),
            None
        )

    # DEPARTMENT

    result = search_department(question, memory)

    if result:

        context = f"""
Department: {result[0]}
{_department_level_line(result)}
Programs:
{result[1]}

Description:
{result[2]}
"""

        # Department coordinator (Sprint 10E) - only added when
        # specifically asked for, mirroring the PROGRAM coordinator
        # pattern above exactly: reads the existing `coordinator` column
        # directly, never inferred from the free-text description, and
        # every other department query keeps producing exactly the
        # context above, unchanged.
        if _has_coordinator_intent(question_lower):

            coordinator = result["coordinator"]

            context += (
                f"\nDepartment Coordinator:\n{coordinator.strip()}\n"
                if coordinator and coordinator.strip() else
                "\nDepartment Coordinator: Coordinator information is not available.\n"
            )

        return context, result[3]

    # FACULTY

    result = search_faculty(question, memory)

    if result:

        context = f"""
Name: {result[0]}
Title: {result[1]}
Faculty: {result[2]}
Department: {result[3]}

Contact:
Email: {result[4]}
Phone: {result[5]}
Office: {result[6]}

Biography:
{result[8]}

Research Interests:
{result[7]}
"""

        return context, result[9]

    # RESEARCH TOPIC
    # Tried last, after every exact/structured lookup above (course,
    # program, coordinator, faculty-level list, department-level list,
    # department, single-person) - this is the least specific signal in
    # the cascade and must never preempt any of those.

    research_topic_result = search_faculty_by_research_topic(question, memory)

    if research_topic_result:

        topic, faculty_rows = research_topic_result

        return (
            _format_faculty_list_context("Research Topic", topic, faculty_rows),
            None
        )

    return None


# Sprint 7A found that falling through to hybrid/vector search on a bare
# pronoun or vague follow-up ("Does it have prerequisites?", "Who
# teaches it?") reliably produces a confident, fabricated answer - the
# LLM has just enough chat history to sound plausible, but no actual
# retrieved data to ground it. resolve_contextual_reference() is a
# deterministic gate against exactly that: called only after
# structured_search() has already failed on the raw question, it checks
# for a contextual-reference marker and, using ONLY the existing 4-slot
# memory, either substitutes it with a real entity and re-attempts
# structured_search, or - if nothing in memory resolves it - returns a
# clarification instead of ever reaching hybrid_search.
# "Compare them/those/these/it" still always clarifies (Sprint 9B) - not
# because the entities can't be resolved anymore (entity_history often
# can identify them now), but because no comparison FEATURE exists for
# any entity type. Resolution and capability are separate concerns: this
# pattern is a capability gap, not a resolution gap, so it's kept apart
# from the (now-resolvable) ordinal pattern below.
_COMPARE_PATTERN = re.compile(
    r"\bcompare\s+(?:them|those|these|it)\b", re.IGNORECASE
)

# Ordinal references ("the first one", "the second one", ...) - Sprint 9B
# adds real resolution for these via entity_history's list_id/
# list_position fields, which the four-slot dict had no way to
# represent at all (hence why these always clarified before).
_ORDINAL_PATTERN = re.compile(
    r"\bthe\s+(first|second|third|fourth|fifth|last)\s+one\b",
    re.IGNORECASE
)

_ORDINAL_POSITIONS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}

# Multi-word phrases that name the entity type explicitly - only that
# one memory slot is ever tried, since the phrase itself is specific.
_TYPE_HINTED_PATTERNS = [
    (re.compile(r"\bthat professor\b", re.IGNORECASE), "faculty"),
    (re.compile(r"\bthe professor\b", re.IGNORECASE), "faculty"),
    (re.compile(r"\bthat course\b", re.IGNORECASE), "course"),
    (re.compile(r"\bthe course\b", re.IGNORECASE), "course"),
    (re.compile(r"\bthat department\b", re.IGNORECASE), "department"),
    (re.compile(r"\bthe department\b", re.IGNORECASE), "department"),
    (re.compile(r"\bthat program\b", re.IGNORECASE), "program"),
    (re.compile(r"\bthe program\b", re.IGNORECASE), "program"),
]

# "they"/"them" most commonly refer to a person in English, so faculty
# is tried first, falling back to the standard priority below only if
# no faculty is on record.
_PERSON_HINTED_PATTERNS = [
    re.compile(r"\bthey\b", re.IGNORECASE),
    re.compile(r"\bthem\b", re.IGNORECASE),
]

# Bare, type-agnostic references - tried against each memory slot in the
# same priority order structured_search() already uses for the
# follow-up-phrase mechanism.
_GENERIC_REFERENCE_PATTERNS = [
    re.compile(r"\bit\b", re.IGNORECASE),
    re.compile(r"\bits\b", re.IGNORECASE),
    re.compile(r"\bthat\b", re.IGNORECASE),
    re.compile(r"\bthis\b", re.IGNORECASE),
    re.compile(r"\bthose\b", re.IGNORECASE),
    re.compile(r"\bthese\b", re.IGNORECASE),
]

_DEFAULT_TYPE_PRIORITY = ["course", "program", "department", "faculty"]
_PERSON_TYPE_PRIORITY = ["faculty", "course", "program", "department"]

_CLARIFICATION_MESSAGES = {
    "course": "I'm not sure which course you mean. Could you mention the course code or name?",
    "program": "I'm not sure which program you mean. Could you mention the program name?",
    "department": "I'm not sure which department you mean. Could you mention the department name?",
    "faculty": "I'm not sure which professor you're referring to. Could you mention their name?",
}

_GENERIC_CLARIFICATION_MESSAGE = (
    "I'm not sure what you're referring to. Could you clarify or "
    "provide a bit more detail?"
)


# Bare pattern.sub() substitution alone isn't reliable: search_course()
# accepts a course code found ANYWHERE in the text, so substituting
# "it" -> "CP312" in "Does it have prerequisites?" produces "Does CP312
# have prerequisites?", which search_course() happily "matches" - but
# only with the general course description, not the prerequisite text
# the user actually asked for. Treating that as a successful resolution
# would just relocate the hallucination risk (the LLM would still guess
# at prerequisites from a context that doesn't contain them). These
# rules are checked first and, when the original question's own wording
# names a specific capability, rewrite directly into the exact phrasing
# that capability's deterministic pattern expects - so a match only
# counts as resolved when it's actually the right feature answering.
_INTENT_REWRITE_RULES = [
    (re.compile(r"\bprerequisites?\b", re.IGNORECASE), "course", "What are the prerequisites for {value}?"),
    (re.compile(r"\bteach(?:es|ing|er)?\b", re.IGNORECASE), "course", "Who has taught {value}?"),
]

# "coordinat..." isn't a fixed-type rule like the two above: before
# Sprint 10E, program coordinator lookup was the only kind that existed,
# so "who coordinates it?" could safely always rewrite toward a program.
# Now that department coordinator lookup also exists, that assumption
# would silently misroute "who coordinates it?" after a department was
# established (e.g. "Tell me about the History department" -> "Who
# coordinates it?") toward a program clarification instead of answering
# from department context. Resolved dynamically instead: whichever of
# program/department was established MORE RECENTLY wins.
_COORDINATOR_REWRITE_PATTERN = re.compile(r"\bcoordinat\w*\b", re.IGNORECASE)

_COORDINATOR_REWRITE_TEMPLATES = {
    "program": "Who is the program coordinator for {value}?",
    # Deliberately phrased as a full sentence (not just the bare
    # department name) - search_department()'s single-word academic-
    # signal guard (Sprint 5C) requires an academic-context word
    # alongside a single-word name like "History", which a bare name
    # substitution would otherwise strip away.
    "department": "Who coordinates the {value} department?",
}


def _resolve_coordinator_target(memory):

    candidates = [
        entry for entry in (
            _latest_entity_of_type(memory, "program"),
            _latest_entity_of_type(memory, "department"),
        )
        if entry
    ]

    if candidates:
        candidates.sort(key=lambda e: e["turn_number"], reverse=True)
        best = candidates[0]
        return best["entity_type"], best["display_name"]

    # No entity-history entry for either type - fall back to the legacy
    # program slot only (department has no legacy slot to fall back to).
    legacy_program = memory.get("last_program")

    return ("program", legacy_program) if legacy_program else None


def _attempt_contextual_resolution(question, pattern, type_priority, memory):

    if _COORDINATOR_REWRITE_PATTERN.search(question):

        target = _resolve_coordinator_target(memory)

        if not target:
            return ("clarify", _GENERIC_CLARIFICATION_MESSAGE)

        entity_type, value = target

        rewritten_question = _COORDINATOR_REWRITE_TEMPLATES[entity_type].format(value=value)

        result = structured_search(rewritten_question, memory)

        if result:
            context, source = result
            return ("resolved", context, source)

        return ("clarify", _CLARIFICATION_MESSAGES[entity_type])

    for rule_pattern, rule_type, template in _INTENT_REWRITE_RULES:

        if not rule_pattern.search(question):
            continue

        value = _resolve_typed_value(memory, rule_type)

        if not value:
            return ("clarify", _CLARIFICATION_MESSAGES[rule_type])

        rewritten_question = template.format(value=value)

        result = structured_search(rewritten_question, memory)

        if result:
            context, source = result
            return ("resolved", context, source)

        return ("clarify", _CLARIFICATION_MESSAGES[rule_type])

    for entity_type in type_priority:

        value = _resolve_typed_value(memory, entity_type)

        if not value:
            continue

        # Generic fallback for questions with no specific-capability
        # keyword above: substitutes only the matched marker, preserving
        # the rest of the sentence, so the result still has to actually
        # match an existing structured pattern to count as resolved.
        # Department values get a "department" qualifier appended
        # (Sprint 10E) for the same single-word academic-signal reason
        # as the coordinator template above - "Tell me more about
        # them." -> "Tell me more about History." would otherwise be
        # indistinguishable from a non-WLU use of the word "History".
        search_value = (
            f"{value} department" if entity_type == "department" else value
        )

        substituted_question = pattern.sub(search_value, question, count=1)

        result = structured_search(substituted_question, memory)

        if result:
            context, source = result
            return ("resolved", context, source)

        return ("clarify", _CLARIFICATION_MESSAGES[entity_type])

    return ("clarify", _GENERIC_CLARIFICATION_MESSAGE)


def _resolve_ordinal_entity(memory, position_word):
    """The entry at a given position within the MOST RECENT list-shaped
    result (memory['_last_list_id']) - independent of entity_history scan
    order, since ordinal references are about position within a list,
    not general recency."""

    if memory is None:
        return None

    list_id = memory.get("_last_list_id")

    if not list_id:
        return None

    entries = _entities_in_list(memory, list_id)

    if not entries:
        return None

    if position_word == "last":
        return entries[-1]

    index = _ORDINAL_POSITIONS.get(position_word)

    if index is None or index > len(entries):
        return None

    return entries[index - 1]


def _attempt_ordinal_resolution(question, position_word, memory):

    entry = _resolve_ordinal_entity(memory, position_word)

    if not entry:
        return ("clarify", _GENERIC_CLARIFICATION_MESSAGE)

    # display_name, not entity_id - see _resolve_typed_value()'s
    # docstring: entity_id is a stable key (e.g. a faculty source_url),
    # not necessarily valid text to substitute back into a question.
    value = entry["display_name"]
    entity_type = entry["entity_type"]

    # Only apply an intent-rewrite rule if it targets the SAME type the
    # ordinal resolved to (e.g. "prerequisites" only makes sense applied
    # to a resolved course) - unlike the pronoun path above, an ordinal's
    # type is already fixed by which list it came from, so a mismatched
    # rule must fall through to generic substitution instead of forcing
    # the wrong kind of lookup.
    for rule_pattern, rule_type, template in _INTENT_REWRITE_RULES:

        if rule_type != entity_type or not rule_pattern.search(question):
            continue

        rewritten_question = template.format(value=value)

        result = structured_search(rewritten_question, memory)

        if result:
            context, source = result
            return ("resolved", context, source)

        return ("clarify", _CLARIFICATION_MESSAGES.get(entity_type, _GENERIC_CLARIFICATION_MESSAGE))

    substituted_question = _ORDINAL_PATTERN.sub(value, question, count=1)

    result = structured_search(substituted_question, memory)

    if result:
        context, source = result
        return ("resolved", context, source)

    return ("clarify", _CLARIFICATION_MESSAGES.get(entity_type, _GENERIC_CLARIFICATION_MESSAGE))


def _compare_clarification(memory):

    list_id = memory.get("_last_list_id") if memory else None

    if list_id:

        entries = _entities_in_list(memory, list_id)

        if entries:
            names = ", ".join(e["display_name"] for e in entries[:5])
            # Every clarification message in this project starts with
            # "I'm not sure" - it's the literal substring
            # is_clarification_response() (evaluate.py) checks for, so
            # this one keeps the same prefix instead of introducing a
            # differently-worded message that would silently stop being
            # recognized as a clarification.
            return (
                "clarify",
                f"I'm not sure how to compare these directly, but I can "
                f"tell you about them individually: {names}."
            )

    return ("clarify", _GENERIC_CLARIFICATION_MESSAGE)


def _memory_has_any_context(memory):

    # Legacy four-slot dict first (cheap, and covers every memory dict
    # built the old way, e.g. plain literals in tests) - entity_history
    # second, since several Sprint 9B write-backs (department->faculty
    # list, research-topic list, courses-taught) populate ONLY
    # entity_history and never touch the four legacy slots at all, so
    # relying on the legacy check alone would wrongly treat that context
    # as "empty".
    if any(memory.get(key) for key in _MEMORY_KEY_BY_TYPE.values()):
        return True

    return bool(memory.get("entity_history"))


def resolve_contextual_reference(question, memory=None):

    if memory is None:
        memory = {}

    # If nothing has ever been established in this conversation, a bare
    # reference word is far more likely to be ordinary English grammar
    # inside an unrelated sentence ("What is the philosophy behind this
    # decision?", "This is just common sense psychology.") than a
    # genuine follow-up to a prior answer - confirmed directly: without
    # this guard, standalone sentences that merely contain "this"/"that"
    # as normal phrasing were being intercepted even though there was no
    # established context for them to be following up on. With nothing
    # in memory, this returns None so the existing routing (off-topic
    # gate, then hybrid search) handles the question exactly as it did
    # before this feature existed.
    if not _memory_has_any_context(memory):
        return None

    question_lower = question.lower()

    if _COMPARE_PATTERN.search(question_lower):
        return _compare_clarification(memory)

    ordinal_match = _ORDINAL_PATTERN.search(question_lower)

    if ordinal_match:
        return _attempt_ordinal_resolution(question, ordinal_match.group(1), memory)

    for pattern, entity_type in _TYPE_HINTED_PATTERNS:

        if pattern.search(question_lower):
            return _attempt_contextual_resolution(
                question, pattern, [entity_type], memory
            )

    for pattern in _PERSON_HINTED_PATTERNS:

        if pattern.search(question_lower):
            return _attempt_contextual_resolution(
                question, pattern, _PERSON_TYPE_PRIORITY, memory
            )

    for pattern in _GENERIC_REFERENCE_PATTERNS:

        if pattern.search(question_lower):
            return _attempt_contextual_resolution(
                question, pattern, _DEFAULT_TYPE_PRIORITY, memory
            )

    return None


def hybrid_search(question, memory=None):

    result = structured_search(question, memory)

    if result:
        return result

    # VECTOR SEARCH

    results = search_vector(
        question
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    source = results[
        "metadatas"
    ][0][0]["url"]

    return context, source