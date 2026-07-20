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

    if memory is not None and question_lower in FOLLOWUP_PHRASES:

        if memory.get("last_course"):
            question = memory["last_course"]

        elif memory.get("last_program"):
            question = memory["last_program"]

        elif memory.get("last_department"):
            question = memory["last_department"]

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