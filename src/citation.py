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
import re
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


# Reused from benchmark_runner.py's own _NEGATED_INFO_AVAILABILITY_
# PATTERN (a negation next to a word about information being present/
# documented/stated), already tuned there across many real grounded
# answers - the 0-2 word gap and excluding "cover(s/ed)" are both
# deliberate: a wider gap or "cover" both let the pattern reach across
# an unrelated clause and false-positive on a genuinely confident,
# correct answer ("CP312 does not require any prerequisites and covers
# algorithm design.").
_INFO_AVAILABILITY_WORDS = (
    r"contain(?:s|ed|ing)?|include[sd]?|specify|specifie[sd]|"
    r"define[sd]?|outline[sd]?|mention(?:s|ed)?|detail(?:s|ed)?|"
    r"provide[sd]?|address(?:es|ed)?|state[sd]?|"
    r"indicate[sd]?|available|found|information|details|data|specifics?"
)
_NEGATED_INFO_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|did\s+not|didn't|"
    r"cannot|can't|unable\s+to|no)\b"
    rf"(?:\s+\w+){{0,2}}\s+(?:{_INFO_AVAILABILITY_WORDS})\b",
    re.IGNORECASE
)

# A second, narrower phrasing family the negation pattern above doesn't
# cover: the LLM explicitly says the retrieved content is ABOUT
# something else, rather than saying information is simply missing -
# e.g. "the retrieved information relates to academic misconduct rather
# than coursework programs". Confirmed live: generate_answer()'s vector-
# grounded path cited an unrelated academic-misconduct policy page for
# "course work?" (asked right after establishing a graduate program as
# context) - a false-positive keyword/embedding match ("coursework" the
# word appears in both the misconduct policy and the actual program
# page, in unrelated senses) that the LLM sometimes, but not reliably,
# flags in its own answer text.
_RELATES_TO_OTHER_TOPIC_PATTERN = re.compile(
    r"\brelate[sd]?\s+to\b[^.?!]{0,80}\brather\s+than\b"
    r"|\bpertains?\s+to\b[^.?!]{0,80}\b(?:not|rather\s+than)\b"
    r"|\bnot\s+(?:related|relevant)\s+to\b"
    r"|\b(?:doesn't|does\s+not)\s+(?:relate|pertain)\s+to\b",
    re.IGNORECASE
)


def answer_disclaims_relevance(answer_text):
    """True if the LLM's own grounded answer indicates the retrieved
    source doesn't actually address the question, rather than
    confidently answering it - a citation must never be shown in this
    case, since the underlying source isn't actually authoritative for
    what was asked (see build_citation()'s answer_text parameter).

    A safety net, not the primary fix: confirmed live across repeated
    identical queries that the LLM (temperature=0.7) doesn't reliably
    self-disclaim even when the cited source genuinely doesn't answer
    the question - it will often confidently discuss the tangentially-
    related retrieved content without ever hedging. The actual fix for
    that failure mode is upstream in retriever.py (resolving vague,
    context-dependent follow-ups against established conversation
    memory before they ever reach vector search); this only catches
    the subset of cases where the model does hedge."""

    return bool(
        _NEGATED_INFO_AVAILABILITY_PATTERN.search(answer_text)
        or _RELATES_TO_OTHER_TOPIC_PATTERN.search(answer_text)
    )


def build_citation(source, response_type=None, answer_text=None):
    """source: a URL string, an iterable of URL strings, or None/falsy
    (nothing to cite - every non-factual response already passes source
    as None today, e.g. greetings/clarifications/not_found, and this
    returns None right back for those, unchanged from before this
    module existed).

    answer_text: when given, and the answer's own text indicates the
    retrieved source doesn't actually address the question (see
    answer_disclaims_relevance() above), returns None instead of citing
    a source the answer itself says isn't a good match - never shown as
    if it were authoritative just because retrieval happened to return
    something. Optional and defaults to None (skip the check) since
    most callers pass a response_type this check doesn't apply to
    anyway (deterministic response types were never LLM-phrased to
    begin with).

    Returns None, or:
        {"date": "<display date>",
         "sources": [{"title": <str or None>, "url": <str>}, ...]}

    Exactly one retrieval date regardless of how many sources are
    listed (multiple sources are supported here even though no current
    retriever.py response type produces more than one)."""

    if not source:
        return None

    if answer_text and answer_disclaims_relevance(answer_text):
        return None

    source_urls = [source] if isinstance(source, str) else [s for s in source if s]

    if not source_urls:
        return None

    sources = [
        {"title": _resolve_title(url, response_type), "url": url}
        for url in source_urls
    ]

    return {"date": _retrieval_date(), "sources": sources}
