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