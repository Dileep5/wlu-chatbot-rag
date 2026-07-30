"""Citation enrichment (Phase 3).

Given the bare source URL(s) that structured_search()/hybrid_search()/
resolve_contextual_reference() already returned, resolves the source
page's title and the corpus-wide retrieval date, producing the enriched
shape renderer.py's citation footer displays.

Deliberately a separate, downstream module: it's called from app.py
*after* retrieval has already finished and already decided on a source.
It never touches retriever.py - no retrieval, ranking, structured-query,
or hallucination-gate logic lives here or is affected by this module.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_METADATA_FILE = BASE_DIR / "data" / "corpus_metadata.json"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# response_type -> (db_path, table, name_column) for a direct
# "SELECT <name_column> ... WHERE source_url = ?" title lookup. Each
# entry mirrors a column retriever.py already selects and returns as
# that response's own `source` value (e.g. search_course()'s
# `result[5]`, search_faculty()'s `result[9]`) - this module only reads
# it back by URL; it never changes what any retriever.py function
# selects or returns.
_STRUCTURED_TITLE_LOOKUP = {
    "course": ("data/courses.db", "courses", "course_name"),
    "program": ("data/programs.db", "programs", "program_name"),
    "department_profile": ("data/departments.db", "departments", "department_name"),
    "faculty_profile": ("data/faculty.db", "faculty", "name"),
    "policy": ("data/policies.db", "policies", "policy_title"),
    # search_faculty_courses_taught() (retriever.py) now cites each
    # instructor's own faculty.db source_url, same table/column as
    # faculty_profile above - so a "who teaches X" citation link shows
    # the instructor's name, not a bare URL.
    "course_instructors": ("data/faculty.db", "faculty", "name"),
    # Same fix, same reasoning, for the three sibling "list of faculty"
    # response types (search_faculty_by_faculty_name/_by_department/
    # _by_research_topic, retriever.py) - each cites multiple source_urls
    # at once (build_citation() already resolves a list of URLs one at a
    # time), so every entry in the list independently looks up its own
    # person's name via this same table/column, never a shared/wrong one.
    "faculty_list": ("data/faculty.db", "faculty", "name"),
    "department_faculty_list": ("data/faculty.db", "faculty", "name"),
    "research": ("data/faculty.db", "faculty", "name"),
}

# "coordinator" is used for both program- and department-coordinator
# answers (retriever.py tags both the same way) - tried in this order
# since the response_type alone doesn't say which table the URL
# belongs to.
_COORDINATOR_TITLE_LOOKUP = [
    ("data/programs.db", "programs", "program_name"),
    ("data/departments.db", "departments", "department_name"),
]


def _lookup_structured_title(db_path, table, column, source_url):

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {column} FROM {table} WHERE source_url = ? LIMIT 1",
            (source_url,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _lookup_vector_title(source_url):
    """Vector-sourced citations (response_type == "vector" or unmapped)
    look their title up from the same ChromaDB metadata build_vector_db.py
    already stores per chunk - no new storage, just reading it back."""

    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        collection = client.get_collection("wlu_chatbot_chunks")
        result = collection.get(
            where={"url": source_url}, limit=1, include=["metadatas"]
        )
        metadatas = result.get("metadatas") or []
        return metadatas[0].get("title") if metadatas else None
    except Exception:
        return None


def _resolve_title(source_url, response_type):

    if response_type == "coordinator":

        for db_path, table, column in _COORDINATOR_TITLE_LOOKUP:

            title = _lookup_structured_title(db_path, table, column, source_url)

            if title:
                return title

        return None

    lookup = _STRUCTURED_TITLE_LOOKUP.get(response_type)

    if lookup:
        db_path, table, column = lookup
        return _lookup_structured_title(db_path, table, column, source_url)

    return _lookup_vector_title(source_url)


_DATE_DISPLAY_FORMAT = "%B %d, %Y"


def _retrieval_date():
    """The corpus-wide retrieval date - written automatically by
    crawler.py at the start of every ingestion run (data/corpus_metadata.
    json), never hardcoded here. Falls back to today's date only when no
    such timestamp exists yet (e.g. a data/ directory produced before
    this feature existed) - the one explicit exception this requirement
    allows for using the system clock."""

    try:
        with CORPUS_METADATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        retrieved_at = datetime.fromisoformat(data["retrieved_at"])

        return retrieved_at.strftime(_DATE_DISPLAY_FORMAT)

    except Exception:
        return datetime.now(timezone.utc).strftime(_DATE_DISPLAY_FORMAT)


def build_citation(source, response_type=None):
    """source: a URL string, an iterable of URL strings, or None/falsy
    (nothing to cite - every non-factual response already passes source
    as None today, e.g. greetings/clarifications/not_found, and this
    returns None right back for those, unchanged from before this
    module existed).

    Returns None, or:
        {"date": "<display date>",
         "sources": [{"title": <str or None>, "url": <str>}, ...]}

    Exactly one retrieval date regardless of how many sources are
    listed (multiple sources are supported here even though no current
    retriever.py response type produces more than one)."""

    if not source:
        return None

    source_urls = [source] if isinstance(source, str) else [s for s in source if s]

    if not source_urls:
        return None

    sources = [
        {"title": _resolve_title(url, response_type), "url": url}
        for url in source_urls
    ]

    return {"date": _retrieval_date(), "sources": sources}
