import os
import sqlite3
import re
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
        {level_column}
    FROM courses
    WHERE course_code=?
    """, (course_code,))

    result = cursor.fetchone()

    conn.close()

    if result and memory is not None:
        memory["last_course"] = course_code

    return result


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

    cursor = conn.cursor()

    level_column = ", level" if PROGRAMS_HAVE_LEVEL else ""

    cursor.execute(f"""
    SELECT
        program_name,
        admission_requirements,
        program_requirements,
        source_url
        {level_column}
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

            return row

    return None


def search_department(question, memory=None):

    conn = sqlite3.connect(
        "data/departments.db"
    )

    cursor = conn.cursor()

    level_column = ", level" if DEPARTMENTS_HAVE_LEVEL else ""

    cursor.execute(f"""
    SELECT
        department_name,
        programs,
        description,
        source_url
        {level_column}
    FROM departments
    """)

    rows = cursor.fetchall()

    conn.close()

    question = question.lower()

    for row in rows:

        if row[0].lower() in question:

            if memory is not None:
                memory["last_department"] = row[0]

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
        "SELECT name, title FROM faculty WHERE department_name LIKE ? ORDER BY name",
        (f"%{matched_segment}%",)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return None

    return matched_segment, rows


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
        "SELECT name, title FROM faculty WHERE faculty_name LIKE ? ORDER BY name",
        (f"%{matched_segment}%",)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return None

    return matched_segment, rows


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

                return row

    return None


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

    # COURSE

    result = search_course(question, memory)

    if result:

        context = f"""
Course Code: {result[0]}
Course Name: {result[1]}
Credits: {result[2]}
Department: {result[4]}
{_level_line(result, 6)}
Description:
{result[3]}
"""

        return context, result[5]

    # PROGRAM

    result = search_program(question, memory)

    if result:

        context = f"""
Program: {result[0]}
{_level_line(result, 4)}
Admission Requirements:
{result[1]}

Program Requirements:
{result[2]}
"""

        return context, result[3]

    # FACULTY-LEVEL LIST (e.g. "Faculty of Science", "Faculty of Arts")
    # Tried before the department-level list check: these are a small,
    # fixed set of known institutional names matched deterministically by
    # exact word-set comparison against faculty_name, not the
    # substring/residual matching department names use.

    faculty_level_result = search_faculty_by_faculty_name(question, memory)

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
    # generic description.

    dept_list_result = search_faculty_by_department(question, memory)

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
{_level_line(result, 4)}
Programs:
{result[1]}

Description:
{result[2]}
"""

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