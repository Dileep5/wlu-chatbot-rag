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

    cursor.execute("""
    SELECT
        course_code,
        course_name,
        credits,
        description,
        department_name
    FROM courses
    WHERE course_code=?
    """, (course_code,))

    result = cursor.fetchone()

    conn.close()

    if result and memory is not None:
        memory["last_course"] = course_code

    return result


def search_program(question, memory=None):

    conn = sqlite3.connect(
        "data/programs.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        program_name,
        admission_requirements,
        program_requirements
    FROM programs
    """)

    rows = cursor.fetchall()

    conn.close()

    question = question.lower()

    for row in rows:

        if row[0].lower() in question:

            if memory is not None:
                memory["last_program"] = row[0]

            return row

    return None


def search_department(question, memory=None):

    conn = sqlite3.connect(
        "data/departments.db"
    )

    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        department_name,
        programs,
        description
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


def hybrid_search(question, memory=None):

    # FOLLOWUP MEMORY

    question_lower = question.lower()

    if memory is not None and question_lower in [
        "tell me more",
        "more",
        "explain",
        "details",
        "more details",
        "what about this",
        "what about it"
    ]:

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

Description:
{result[3]}
"""

        return context, "courses.db"

    # PROGRAM

    result = search_program(question, memory)

    if result:

        context = f"""
Program: {result[0]}

Admission Requirements:
{result[1]}

Program Requirements:
{result[2]}
"""

        return context, "programs.db"

    # DEPARTMENT

    result = search_department(question, memory)

    if result:

        context = f"""
Department: {result[0]}

Programs:
{result[1]}

Description:
{result[2]}
"""

        return context, "departments.db"

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