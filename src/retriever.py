import os
import sqlite3
import re
from collections import deque
import chromadb
import streamlit as st
import streamlit.runtime as st_runtime
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer

import hybrid_rerank

DB_DIR = "data/vector_db"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(
    path=DB_DIR
)


# Production deployment fix: data/vector_db/ is a large (~95MB),
# fast-growing generated artifact deliberately excluded from git
# (.gitignore) - rebuilt locally/in Docker from the ingestion pipeline,
# never committed. A fresh git clone (e.g. Streamlit Community Cloud,
# which only ever sees the git-tracked tree, never a machine's local
# disk state) has no vector_db directory at all - chromadb.
# PersistentClient() silently creates an empty one on first use, so the
# unconditional get_collection() call below used to raise
# NotFoundError and crash the app before a single query could ever be
# served, even though nothing was actually wrong with retrieval itself.
#
# outputs/chunks.csv (the same source build_vector_db.py's own offline
# pipeline reads - 4,066 real scraped-and-chunked WLU pages, ~7MB of
# plain text) and data/faculty.db ARE committed, unlike the 95MB binary
# vector_db/ itself, so both collections below can be rebuilt from
# real, already-ingested data on first run whenever they're missing -
# never placeholder/dummy content, and never touching how a query is
# answered once the collection exists. Reimplemented here (not by
# importing build_vector_db.py, which needs pandas) with the stdlib
# csv module instead, so no new runtime dependency is introduced.
def _rebuild_chunks_collection(chroma_client, embedding_model):

    import csv

    print(
        "wlu_chatbot_chunks collection not found - rebuilding from "
        "outputs/chunks.csv (expected on a fresh deploy; one-time cost)..."
    )

    with open("outputs/chunks.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    try:
        chroma_client.delete_collection("wlu_chatbot_chunks")
    except Exception:
        pass

    rebuilt = chroma_client.get_or_create_collection(name="wlu_chatbot_chunks")

    rebuilt.add(
        ids=[row["chunk_id"] for row in rows],
        documents=[row["chunk_text"] for row in rows],
        metadatas=[
            {"title": row["title"], "url": row["url"]} for row in rows
        ],
        embeddings=[
            embedding_model.encode(row["chunk_text"]).tolist() for row in rows
        ],
    )

    print(f"Rebuilt wlu_chatbot_chunks: {len(rows)} chunks.")

    return rebuilt


def _rebuild_faculty_research_collection(chroma_client, embedding_model):
    """Mirrors build_faculty_vector_db.py's own logic exactly, from the
    same committed source (data/faculty.db)."""

    print(
        "wlu_faculty_research collection not found - rebuilding from "
        "data/faculty.db (expected on a fresh deploy; one-time cost)..."
    )

    conn = sqlite3.connect("data/faculty.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT source_url, name, research_interests
    FROM faculty
    WHERE research_interests IS NOT NULL
      AND TRIM(research_interests) != ''
    """)

    rows = cursor.fetchall()

    conn.close()

    try:
        chroma_client.delete_collection("wlu_faculty_research")
    except Exception:
        pass

    rebuilt = chroma_client.get_or_create_collection(name="wlu_faculty_research")

    rebuilt.add(
        ids=[source_url for source_url, _, _ in rows],
        documents=[interests for _, _, interests in rows],
        metadatas=[
            {"source_url": source_url, "name": name}
            for source_url, name, _ in rows
        ],
        embeddings=[
            embedding_model.encode(interests).tolist()
            for _, _, interests in rows
        ],
    )

    print(f"Rebuilt wlu_faculty_research: {len(rows)} records.")

    return rebuilt


def _update_status(status, label, state):
    """status.update() guarded against `status` being None - st.status()
    itself never raises outside a real Streamlit script run (confirmed
    live: evaluation/generate_benchmark.py and a bare `import retriever`
    both reach this code with no script context), but the object it
    yields there IS None, and calling .update() on that raises
    AttributeError - exactly the kind of new, confusing crash this
    deployment-polish feature exists to prevent, not introduce."""

    if status is not None:
        status.update(label=label, state=state)


try:
    _chunks_collection_exists = bool(client.get_collection("wlu_chatbot_chunks"))
except Exception:
    _chunks_collection_exists = False

if _chunks_collection_exists:

    # Normal path (local dev, Docker, or any redeploy after the first
    # one on a given host) - byte-identical to before this file gained
    # first-launch status messaging. No status widget: this is near-
    # instant, not the "first launch" experience the messaging below is
    # for, so nothing should flash on screen for it.
    collection = client.get_collection("wlu_chatbot_chunks")

    # The faculty-research collection is built by the separate
    # build_faculty_vector_db.py pipeline (Sprint 4C1) and may not exist
    # yet in every environment - detected once, the same way
    # FACULTY_DB_READY guards faculty.db access, so its absence degrades
    # gracefully instead of raising. Silent rebuild-on-missing (no
    # status widget, same reasoning as above): this is the rarer
    # partial-init case (chunks present, faculty-research alone
    # missing), not the fresh-deploy case the messaging below covers.
    try:
        faculty_research_collection = client.get_collection(
            "wlu_faculty_research"
        )
        FACULTY_RESEARCH_READY = True
    except Exception:
        try:
            faculty_research_collection = _rebuild_faculty_research_collection(
                client, model
            )
            FACULTY_RESEARCH_READY = True
        except Exception:
            faculty_research_collection = None
            FACULTY_RESEARCH_READY = False

else:

    # First-launch path (e.g. a fresh Streamlit Community Cloud clone,
    # see _rebuild_chunks_collection()'s own comment) - both collections
    # are rebuilt together under one visible status message, since a
    # missing chunks collection means the sibling faculty-research
    # collection living in the same vector_db/ directory is, in
    # practice, missing too.
    with st.status(
        "Preparing the knowledge base for first launch. This may take "
        "1-2 minutes.",
        expanded=True,
    ) as _status:

        try:
            collection = _rebuild_chunks_collection(client, model)

            try:
                faculty_research_collection = (
                    _rebuild_faculty_research_collection(client, model)
                )
                FACULTY_RESEARCH_READY = True
            except Exception:
                faculty_research_collection = None
                FACULTY_RESEARCH_READY = False

            _update_status(_status, "Knowledge base ready.", "complete")

        except Exception:

            _update_status(
                _status,
                "Couldn't prepare the knowledge base. Please try "
                "reloading this page in a few minutes, or contact the "
                "site administrator if this keeps happening.",
                "error",
            )

            # No vector-backed query can work without `collection` -
            # nothing left to do but stop. st.stop() is Streamlit's own
            # clean, no-traceback halt (only meaningful with a real
            # script run behind it, confirmed live it's a no-op
            # otherwise), so re-raise instead when there's no Streamlit
            # runtime to catch it (evaluation/generate_benchmark.py, a
            # bare `import retriever`) - there, the friendly status
            # widget above can't render anyway, and silently continuing
            # past this with `collection` never assigned would only
            # replace one clear error with a much more confusing one
            # the first time anything tries to use it.
            if st_runtime.exists():
                st.stop()
            raise

FOLLOWUP_PHRASES = [
    "tell me more",
    "more",
    "explain",
    "details",
    "more details",
    "what about this",
    "what about it",
    # The natural ways someone responds to app.py's answer-first
    # redesign prompt ("Want the full details? Just ask") - added
    # alongside the phrasings above rather than as a separate
    # mechanism, since they need the exact same FOLLOWUP MEMORY
    # rewrite immediately below to mean anything at all: a bare "show
    # me" or "yes please" names no entity of its own, so without this
    # rewrite substituting the established course/program/department/
    # faculty name in for it first, structured_search() would never
    # match anything and the request would fall through unresolved.
    "yes please",
    "show me",
    "show details",
    "show the details",
    "show me more",
    "full details",
    "give me the details",
    "give me the full details",
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

# policies.db doesn't exist until the Phase 2 policy-index load has been
# run - detected the same way every other newer table is, so search_policy
# can no-op gracefully rather than raising in an environment that
# predates it.
POLICIES_DB_READY = _table_exists("data/policies.db", "policies")


# --- Deterministic course-NAME lookup (course-CODE lookup, above the
# class definition below, is unchanged and still tried first) ---
#
# Bare names under this length collide with ordinary English far too
# easily (matches _MIN_NAME_TOKEN_LENGTH's identical role in faculty-
# name matching) - the shortest real course names in the catalog are
# 4-character instrument names ("Oboe", "Tuba"), so this excludes
# nothing legitimate while still guarding against degenerate entries.
_COURSE_NAME_MIN_LENGTH = 4

# The signal that gates a single-word course-name match (see
# _search_course_by_name() below) - deliberately narrower than any
# general WLU-domain keyword list: the query must reference a course as
# a CONCEPT, not merely be WLU-related in some unrelated way (an
# "international student" question, a "program" question, etc. are all
# genuinely WLU-related without being about any specific course).
_COURSE_CONTEXT_HINT_PATTERN = re.compile(
    r"\bcourses?\b|\bclass(?:es)?\b|\bcredits?\b|\bprerequisites?\b|"
    r"\bcorequisites?\b|\bsyllabus\b|\bsection\b|\bexclusions?\b",
    re.IGNORECASE,
)


class _AmbiguousCourseMatch:
    """Sentinel returned by search_course() when a name-based lookup
    matches more than one distinct course (e.g. "Special Topics" is
    reused as a generic title by dozens of different departments) - the
    caller must ask the user to clarify rather than silently picking
    whichever row happened to come back first from the DB."""

    __slots__ = ("candidates",)

    def __init__(self, candidates):
        self.candidates = candidates


def _search_course_by_name(question, memory=None):

    conn = sqlite3.connect(
        "data/courses.db"
    )

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
    WHERE course_name IS NOT NULL
    """)

    rows = cursor.fetchall()

    conn.close()

    question_lower = question.lower()

    # This is the least specific structured signal in the whole cascade
    # (see structured_search()'s "COURSE NAME (last resort)" comment) - a
    # bare substring match against the entire ~4600-row catalog, which
    # includes plenty of one-word titles that are also ordinary dictionary
    # words or general-knowledge topics with zero connection to WLU
    # (GG250 is literally titled "Canada"; others include "Grief",
    # "Poverty", "Evolution", "Optics", "Ecology", "Auditing", "Geometry",
    # "Vikings!"). Confirmed live: "Tell me about Canada" and "What is
    # Poverty?" - genuine out-of-domain questions - matched these titles
    # and returned a WLU course card/clarification instead of being
    # declined. A multi-word title ("Operating Systems", "Machine
    # Learning", "Consumer Behaviour") is a much stronger signal on its
    # own - vanishingly unlikely to appear by coincidence in an unrelated
    # sentence - so only single-word titles are held to the extra bar
    # below; every existing multi-word course-name match is unaffected.
    #
    # QA Fix Sprint regression: originally gated on matches_wlu_keywords()
    # (any of ~80 generic WLU-domain words), which is too broad here -
    # confirmed live, "I am an international student from Canada, what
    # should I know?" matches "international student" (a real WLU
    # keyword) and STILL matched the single-word "Canada" course title,
    # reproducing the exact GG250 clarification bug for a query that has
    # nothing to do with that course. Narrowed to a course-specific
    # signal - the query must actually reference a course as a concept
    # ("course", "class", "credits", "prerequisites", ...), not merely
    # be WLU-related in general - since that's the only thing that
    # actually justifies resolving a bare single word as a course name.
    has_domain_signal = bool(_COURSE_CONTEXT_HINT_PATTERN.search(question))

    # Every distinct course name that appears as a whole-phrase,
    # word-boundary-bounded match in the question - e.g. "Operating
    # Systems" matches both the bare query and "Tell me about Operating
    # Systems", but never matches as a fragment inside an unrelated
    # word.
    matched_names = set()

    for row in rows:

        name = (row["course_name"] or "").strip()

        if len(name) < _COURSE_NAME_MIN_LENGTH:
            continue

        if not has_domain_signal and len(name.split()) == 1:
            continue

        if re.search(rf"\b{re.escape(name.lower())}\b", question_lower):
            matched_names.add(name)

    # Keep only maximal matches: a generic name that's also a substring
    # of a more specific name that ALSO matched (e.g. a course literally
    # named "Thesis" would otherwise also register whenever the question
    # says "Master's Thesis") is strictly less specific and gets
    # dropped, rather than inflating the ambiguity count with a name the
    # user didn't actually type.
    maximal_names = {
        name for name in matched_names
        if not any(
            name != other and name.lower() in other.lower()
            for other in matched_names
        )
    }

    if not maximal_names:
        return None

    candidates = [
        row for row in rows
        if (row["course_name"] or "").strip() in maximal_names
    ]

    if len(candidates) > 1:
        return _AmbiguousCourseMatch(candidates)

    row = candidates[0]

    if memory is not None:
        memory["last_course"] = row[0]
        _record_entity(
            memory, "course", row[0], f"{row[0]} - {row[1]}",
            "search_course",
        )

    return row


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
    ORDER BY CAST(substr(source_url, instr(source_url, 'y=') + 2) AS INTEGER) DESC
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

    # Phase 12C: every path below is already complete, final-answer text
    # (built once, verified across Sprints 7B/9B/9C) - tagged "prerequisite"
    # here, at the single point all four handlers funnel through, rather
    # than touching each handler's own return statements.
    match = _REVERSE_PREREQUISITE_PATTERN.search(question)

    if match:
        result = _handle_reverse_prerequisite_lookup(match.group(1), memory)
        return (*result, "prerequisite") if result else None

    if _NO_PREREQUISITE_PATTERN.search(question):
        result = _handle_no_prerequisite_courses(memory)
        return (*result, "prerequisite") if result else None

    match = _REQUIRES_RELATIONSHIP_PATTERN.search(question)

    if match:
        result = _handle_requires_relationship(
            match.group(1), match.group(2), memory
        )
        return (*result, "prerequisite") if result else None

    match = _DIRECT_PREREQUISITE_PATTERN.search(question)

    if match:
        result = _handle_direct_prerequisite_lookup(match.group(1), memory)
        return (*result, "prerequisite") if result else None

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

# Sprint 11C: undergraduate-style phrasings of the same "list required
# courses" question, with the words in a different order than the
# graduate-derived pattern above expects ("courses required for X"
# rather than "required courses for X", and "what is required for X"
# with no "course(s)" word at all). Both map to the same handler.
_COURSES_REQUIRED_FOR_PATTERN = re.compile(
    r"\bcourses?\s+(?:are\s+|is\s+)?required\s+for\s+(?:the\s+)?(.+)",
    re.IGNORECASE
)
_WHAT_IS_REQUIRED_PATTERN = re.compile(
    r"\bwhat(?:'s|\s+is)\s+required\s+for\s+(?:the\s+)?(.+)",
    re.IGNORECASE
)

# "What courses are there in X?" / "what courses are offered in X?" /
# "what courses does X have/offer?" - a genuinely different phrasing
# family from every pattern above, none of which fire unless the
# question says "required(s)" somewhere. Confirmed live: "what courses
# are there in mac?" matched none of them and fell through the entire
# structured cascade straight to vector search, which had no way to
# know "mac" meant the program already established in conversation
# memory and surfaced an unrelated building/orientation page instead.
# Maps to the same _handle_program_required_courses() handler as the
# "required courses" family - "list the courses" and "list the
# required courses" are the same underlying capability, just phrased
# without the word "required".
_COURSES_IN_PROGRAM_PATTERN = re.compile(
    r"\bwhat\s+courses?\s+(?:are\s+(?:there\s+|offered\s+)?|is\s+there\s+)"
    r"(?:in|at|for)\s+(?:the\s+)?(.+)",
    re.IGNORECASE
)
_PROGRAM_HAS_COURSES_PATTERN = re.compile(
    r"\bwhat\s+courses?\s+does\s+(?:the\s+)?(.+?)\s+(?:have|offer)\b",
    re.IGNORECASE
)

# A query that's ENTIRELY "course work"/"coursework"/"courses" (plus
# trivial punctuation) and nothing else - too generic to have a topic
# of its own, unlike _is_referentless_query()'s pronoun-reference
# queries ("it", "that"), so it's never caught by that guard, but the
# same underlying problem: with no program named, this only means
# anything as a follow-up to an already-established program. Confirmed
# live: after establishing "Master of Applied Computing" as context,
# "course work?" fell through to vector search, which had no way to
# know the established program and confidently cited an unrelated
# academic-misconduct policy page that happens to also use the word
# "coursework" - a false-positive keyword/embedding match, not a real
# answer to what was actually asked. Only ever resolves via memory
# (there's no program name in the query text at all to try first,
# unlike _handle_courses_in_program() above) - with nothing
# established, this still falls through to vector search unprotected,
# same as before this fix; that's a pre-existing, unreported gap this
# bug fix doesn't attempt to also close.
_BARE_COURSEWORK_PATTERN = re.compile(
    r"^\s*(?:course\s*work|courses?)\s*\??\s*$",
    re.IGNORECASE
)

# Year-specific lookup (Sprint 11C) - a genuinely new query shape with
# no graduate equivalent (graduate program_course_requirements has no
# year concept at all). Accepts both numeric ("Year 2") and ordinal-word
# ("first year"/"1st year") phrasing, since real usage naturally mixes
# both ("What courses are in Year 2?" vs "What do I take in first
# year?").
_YEAR_WORD_TO_NUMBER = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
}

_YEAR_REFERENCE_PATTERN = re.compile(
    r"\byear\s+(\d+)\b|\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b",
    re.IGNORECASE
)


def _extract_year_number(text):

    match = _YEAR_REFERENCE_PATTERN.search(text)

    if not match:
        return None

    if match.group(1):
        return int(match.group(1))

    return _YEAR_WORD_TO_NUMBER.get(match.group(2).lower())


def _all_requirement_program_names():

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM program_course_requirements"
    )

    names = [row[0] for row in cursor.fetchall()]

    conn.close()

    return names


def _all_program_names():

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT program_name FROM programs")

    names = [row[0] for row in cursor.fetchall()]

    conn.close()

    return names


def _match_program_name(text, allow_dataless=False):
    """allow_dataless: when the single most specific matching real
    program has no program_course_requirements rows, return its name
    anyway instead of None - only safe for callers that go on to report
    "no structured data available for <name>" (a correct, honest answer
    once <name> is verified as the right program). Left False for every
    other caller, which instead needs "no match" itself to mean "I'm not
    confident which program this is" - e.g. _handle_program_requires_
    course()'s yes/no relationship check falls back to showing the
    plain course card when no program matches, which double-checks a
    program/course RELATIONSHIP claim, not just a program's identity,
    and a fixed regression test (see Sprint 11C/12C's "Undergraduate
    exclusion: no fabricated requirement claim" check) has a permanent
    wording expectation tied to that specific fallback."""

    text_lower = text.lower()

    names = _all_requirement_program_names()

    # Tier 1: exact stored-name substring match (unchanged) - the
    # stored name appears verbatim somewhere in the user's text. Works
    # for graduate names, which users tend to type out close to in full
    # ("Master of Applied Computing").
    for name in sorted(names, key=len, reverse=True):

        if name.lower() in text_lower:
            return name

    # Guard (Sprint 11C): if the user's text already names a DIFFERENT,
    # more specific real program that simply has no structured
    # requirement data of its own (e.g. a joint-program HTML-table page,
    # out of scope this sprint), the bare-subject fallback below must
    # not instead resolve to an unrelated sibling program that only
    # happens to share the same bare subject words. Confirmed live:
    # "the Honours BSc in Computer Science and Honours Bachelor of
    # Business Administration program" was incorrectly resolving to the
    # plain "Honours BSc Computer Science" major, since the joint
    # program (having no requirement rows) was never even a Tier 1
    # candidate, letting Tier 2 match its unrelated sibling instead.
    for name in sorted(_all_program_names(), key=len, reverse=True):

        if name.lower() in text_lower and name not in names:
            return None

    # Tier 2: bare-subject fallback (Sprint 11C) - mirrors
    # search_program()'s own Tier 2b (Sprint 11B). Undergraduate program
    # titles are consistently degree-prefixed ("Honours BSc Computer
    # Science"), but users just name the subject ("Computer Science
    # required courses") - the STORED name is longer than what the user
    # typed, the opposite direction from Tier 1, so the stored name's
    # bare subject (degree prefix stripped) is checked as a substring of
    # the user's text instead. Unlike search_program()'s Tier 2b, no
    # additional single-word signal-word guard is needed here: every
    # caller of _match_program_name() is only ever reached through a
    # regex that already requires "require(s)"/"required" to be present
    # in the original question (_PROGRAM_REQUIRES_COURSE_PATTERN,
    # _PROGRAM_REQUIRED_COURSES_PATTERN and friends) - that's already a
    # strong, unambiguous academic-context signal on its own, so
    # "Biology required courses" resolves without needing a second
    # qualifier the way a bare "Tell me about Biology" would.
    #
    # Restricted to the LONGEST matching subject among ALL real programs
    # (_all_program_names()), not just the ones with requirement data -
    # mirroring search_program()'s own Phase 13F longest-subject-first
    # fix, applied here for the same reason. Without this, a shorter,
    # more generic subject that DOES have requirement rows ("science")
    # can win purely because a longer, more specific subject ("data
    # science") belongs to a program with none - confirmed live: "What
    # courses are required for the Data Science minor?" incorrectly
    # returned Honours BSc Science's requirement list, since "Honours
    # BSc Data Science" (having zero program_course_requirements rows)
    # was never even a Tier 2 candidate under the old names-only loop,
    # leaving "science" free to match as an unopposed substring. Once
    # the single most specific matching subject is identified across
    # every real program, a requirement-data-having candidate at that
    # exact specificity is preferred if one exists; otherwise (only for
    # allow_dataless=True callers) the most specific real program name
    # is still returned - not a shorter, unrelated one - so the
    # caller's own "no structured data available for X" fallback names
    # the CORRECT program instead of silently substituting a different
    # one.
    best_subject_length = 0
    best_candidates = []

    for name in sorted(_all_program_names(), key=len, reverse=True):

        subject = _strip_to_subject(name.lower())

        if not (
            len(subject) >= 3
            and subject in text_lower
            and not _subject_match_degree_conflicts(name, text_lower)
        ):
            continue

        if len(subject) > best_subject_length:
            best_subject_length = len(subject)
            best_candidates = [name]

        elif len(subject) == best_subject_length:
            best_candidates.append(name)

    for name in best_candidates:

        if name in names:
            return name

    if allow_dataless and best_candidates:
        return best_candidates[0]

    return None


# Phase 12C: program_course_requirements holds both graduate (Sprint 7C)
# and undergraduate (Sprint 11C/11E) rows in one table, distinguished by
# `level` - this reads that column directly rather than guessing from
# program_name shape, so the deterministic-response tag always matches
# the data's own real classification.
def _program_requirement_level(program_name):

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT level FROM program_course_requirements "
        "WHERE program_name = ? LIMIT 1",
        (program_name,)
    )

    row = cursor.fetchone()

    conn.close()

    if row and row[0] == "undergraduate":
        return "undergraduate_requirements"

    return "graduate_requirements"


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
            None,
            "graduate_requirements"
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

    # Phase 12C: this reverse lookup's own wording has always said
    # "Graduate programs..." (Sprint 7D), so it's tagged that way
    # unconditionally here too, rather than re-checking each returned
    # program's individual level.
    return (
        f"Graduate programs that require {course_code}:\n{lines}",
        None,
        "graduate_requirements"
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

    response_type = _program_requirement_level(program_name)

    if found:
        return (
            f"Yes, {program_name} lists {course_code} as a required "
            f"course.",
            None,
            response_type
        )

    return (
        f"{course_code} is not listed as a required course for "
        f"{program_name}, based on available structured data. This "
        f"only covers explicitly required courses - electives and "
        f"categorical requirements aren't included.",
        None,
        response_type
    )


def _handle_program_required_courses(program_phrase, memory=None):

    # allow_dataless=True: this handler's own "no structured required-
    # course data is available for X" fallback below is exactly the
    # honest, correct response once X is verified as the right program -
    # unlike _handle_program_requires_course()'s yes/no relationship
    # check, there's no separate claim here that could be wrong if the
    # program has no data of its own.
    program_name = _match_program_name(program_phrase, allow_dataless=True)

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
            None,
            _program_requirement_level(program_name)
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
        None,
        _program_requirement_level(program_name)
    )


def _handle_courses_in_program(program_phrase, memory=None):
    """Wraps _handle_program_required_courses() with a fallback to
    conversation memory when the phrase itself doesn't resolve to a
    real program - scoped to _COURSES_IN_PROGRAM_PATTERN/
    _PROGRAM_HAS_COURSES_PATTERN only, not the older "required courses
    for X" patterns, which stay exactly as strict as before (naming a
    program that doesn't exist there is exactly the "no fabrication"
    case those patterns are already tested against, so falling back to
    a stale memory program for THEM would risk answering about the
    wrong program instead of correctly finding nothing).

    This pattern family is different: "what courses are there in mac?"
    is a genuine follow-up-shaped question, and _match_program_name()
    has no acronym-expansion of its own (unlike search_program()'s
    Tier 2, which only recognizes acronyms typed in uppercase, by
    design, to avoid false positives against ordinary lowercase words -
    a real user typing a lowercase acronym like "mac" would never
    match it either way). Confirmed live: after establishing "Master of
    Applied Computing" as context across several turns, "what courses
    are there in mac?" fell through the entire structured cascade to
    vector search and surfaced an unrelated Milton campus orientation
    page - established context should win over an unrelated vector
    match for exactly this kind of short, ambiguous, already-resolved
    reference."""

    program_name = _match_program_name(program_phrase, allow_dataless=True)

    if not program_name:
        program_name = _resolve_typed_value(memory, "program")

    if not program_name:
        return None

    return _handle_program_required_courses(program_name, memory)


def _handle_program_year_courses(program_name, year, memory=None):

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT course_code, term FROM program_course_requirements "
        "WHERE program_name = ? AND year = ? ORDER BY course_code",
        (program_name, year)
    )

    rows = cursor.fetchall()

    conn.close()

    if not rows:
        _record_entity(
            memory, "program", program_name, program_name,
            "search_program_course_requirements",
        )
        return (
            f"No structured Year {year} course data is available for "
            f"{program_name}.",
            None,
            "undergraduate_requirements"
        )

    _record_entity_list(
        memory, "course", [(code, code) for code, term in rows],
        "search_program_course_requirements",
    )
    _record_entity(
        memory, "program", program_name, program_name,
        "search_program_course_requirements",
    )

    lines = "\n".join(
        f"- {code} ({term})" if term else f"- {code}"
        for code, term in rows
    )

    return (
        f"Year {year} courses listed for {program_name}:\n{lines}\n"
        f"(This reflects explicitly listed required courses only - "
        f"electives and categorical requirements aren't included.)",
        None,
        "undergraduate_requirements"
    )


def search_program_course_requirements(question, memory=None):

    if not PROGRAM_COURSE_REQUIREMENTS_READY:
        return None

    # Year-specific lookup (Sprint 11C), checked first as the more
    # specific shape. Uniquely supports resolving the program from
    # conversation memory when the question doesn't name one at all
    # ("What do I take in first year?" only makes sense as a follow-up
    # to an already-established program) - unlike every other pattern
    # in this function, which requires the program to be named directly.
    year_number = _extract_year_number(question)

    if year_number is not None:

        program_name = _match_program_name(question)

        if not program_name:
            program_name = _resolve_typed_value(memory, "program")

        if program_name:
            result = _handle_program_year_courses(program_name, year_number, memory)
            if result:
                return result

    if _BARE_COURSEWORK_PATTERN.match(question):

        program_name = _resolve_typed_value(memory, "program")

        if program_name:
            result = _handle_program_required_courses(program_name, memory)
            if result:
                return result

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

    match = _COURSES_REQUIRED_FOR_PATTERN.search(question)

    if match:
        result = _handle_program_required_courses(match.group(1), memory)
        if result:
            return result

    match = _WHAT_IS_REQUIRED_PATTERN.search(question)

    if match:
        result = _handle_program_required_courses(match.group(1), memory)
        if result:
            return result

    match = _COURSES_IN_PROGRAM_PATTERN.search(question)

    if match:
        result = _handle_courses_in_program(match.group(1), memory)
        if result:
            return result

    match = _PROGRAM_HAS_COURSES_PATTERN.search(question)

    if match:
        result = _handle_courses_in_program(match.group(1), memory)
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

# Matches a bare reference word as the FIRST word of the captured text,
# not just when the captured text is exactly one of these words alone.
# A compound query like "...and who teaches it and what are the
# prerequisites?" captures "it and what are the prerequisites" here -
# the old exact-equality check against _BARE_REFERENCE_WORDS missed this
# (the captured text isn't *exactly* "it"), so the leading pronoun was
# treated as a literal course-name candidate, which then failed to
# match anything and produced a false "no course found" - even when the
# course (e.g. CP104, named earlier in the same sentence) genuinely
# exists and search_course()'s own code-shape regex would have found it
# moments later in the cascade, if this hadn't already claimed the
# query with a wrong answer. Requiring only a LEADING pronoun (still
# followed by a word boundary, so "them" doesn't wrongly match inside
# "themed") correctly declines this compound case too, letting the
# question fall through to whichever capability actually can answer it.
_BARE_REFERENCE_LEADING_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(w) for w in _BARE_REFERENCE_WORDS) + r")\b",
    re.IGNORECASE
)


def _extract_taught_query(question):

    match = _TAUGHT_INTENT_PATTERN.search(question)

    if not match:
        return None

    captured = match.group(1).strip().rstrip("?.!, ")

    if not captured:
        return None

    if _BARE_REFERENCE_LEADING_PATTERN.match(captured):
        return None

    return captured


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

    # Confirmed live: this branch always returned None here despite
    # `rows` already carrying each instructor's own real profile
    # source_url from the query above - a genuine missing-citation bug,
    # not a hallucination (the underlying faculty_courses_taught/faculty
    # data itself is correct and grounded). citation.py's
    # build_citation() already accepts an iterable of URLs, so every
    # instructor's profile is cited, matching how every other correct
    # structured answer in this app shows a source.
    faculty_source_urls = [source_url for name, title, source_url in rows]

    return (
        _format_faculty_list_context(
            "Faculty who have taught this course", label, display_rows
        ),
        faculty_source_urls
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

    if isinstance(person_row, _AmbiguousFacultyMatch):
        names = sorted({row[0] for row in person_row.candidates})
        return (
            "I'm not sure which professor you mean - I found multiple "
            f"faculty members matching that name: {', '.join(names)}. "
            "Could you provide a full name?",
            None
        )

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
    # Bare "master(s)"/"master's", with no "of" immediately following
    # (already handled by the "master of" rule above, and by then
    # already consumed - this rule only ever reaches whatever "master"
    # occurrences that one didn't). Real graduate program titles always
    # spell out "Master of X" the way "master of" above expects, but
    # real users just as often phrase it "masters in X" or "a master's
    # degree in X" instead - confirmed live: "masters in computer
    # science" left "masters" completely unnormalized, so
    # _subject_match_degree_conflicts() found no degree token in the
    # user's text at all and never blocked the match against the
    # unrelated undergraduate "Honours BSc Computer Science" - the
    # guard exists specifically for this kind of cross-level collision
    # but silently did nothing whenever the user's own phrasing wasn't
    # graduate-title-shaped.
    (r"\bmaster'?s?\b", " mast "),
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

# Capturing variant of the same pattern, used only to extract which
# degree-level token(s) are present (for the cross-level conflict guard
# below) rather than to strip them.
_DEGREE_TOKEN_CAPTURE_PATTERN = re.compile(r"\b(bach|mast|doc|dip)\b")


def _strip_to_subject(text):

    return re.sub(r"\s+", " ", _DEGREE_TOKEN_PATTERN.sub(" ", _strip_filler(text))).strip()


def _mentioned_degree_tokens(text):

    return set(_DEGREE_TOKEN_CAPTURE_PATTERN.findall(_strip_filler(text.lower())))


# "Graduate" (unlike "masters"/"doctor of"/"diploma in") doesn't map to
# one specific degree-level token - a graduate program could be a
# Master's, Doctoral, or Graduate Diploma - so it can't be folded into
# _mentioned_degree_tokens()/the token-disjoint check below without
# also, incorrectly, blocking real doctoral/diploma candidates whenever
# a user just says "graduate". Its role here is narrower and
# unambiguous instead: an undergraduate ("bach") candidate should never
# match when the user's text explicitly says "graduate".
_GRADUATE_LEVEL_SIGNAL_PATTERN = re.compile(r"\bgraduate\b", re.IGNORECASE)


# Bare-subject matching alone loses degree-level information entirely -
# confirmed live during Sprint 11C verification: "Does the Honours
# Bachelor of Business Administration require CP104?" incorrectly
# matched the unrelated graduate "Master of Business Administration"
# program, since both reduce to the identical bare subject "business
# administration". A bare-subject match is rejected whenever the
# CANDIDATE program's own degree-level token is explicitly contradicted
# by a DIFFERENT degree-level token the user's own text mentions -
# "Bachelor"/"BSc"/"BA" in the question should never resolve to a
# "Master of..." program, and vice versa. A question that mentions no
# degree level at all ("Computer Science") stays unaffected - only an
# explicit, conflicting mention blocks the match.
def _subject_match_degree_conflicts(candidate_name, text_lower):

    candidate_tokens = _mentioned_degree_tokens(candidate_name)
    text_tokens = _mentioned_degree_tokens(text_lower)

    if candidate_tokens and text_tokens and candidate_tokens.isdisjoint(text_tokens):
        return True

    if "bach" in candidate_tokens and _GRADUATE_LEVEL_SIGNAL_PATTERN.search(text_lower):
        return True

    return False


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


# Sprint 2: a question that explicitly says "department"/"departments" is
# asking about a DEPARTMENT page, not about a program that happens to share
# the subject's name. This gates the search_program() guard below (see its
# comment), and is deliberately distinct from the broader academic-signal
# pattern used by _department_name_matches().
_DEPARTMENT_INTENT_PATTERN = re.compile(r"\bdepartments?\b", re.IGNORECASE)


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

    # Department-intent guard (Sprint 2): when the question explicitly
    # names a "department" AND search_department() can resolve it, the
    # query is about the DEPARTMENT page - "Tell me about the Ancient
    # Studies department" must return the Ancient Studies department
    # overview, not the "Honours BA Ancient Studies" program whose
    # subject words happen to be a substring of the question. Without
    # this, the program branch (which runs earlier in structured_search()
    # than the department branch) captures every "the X department"
    # question, so the department lookup is never reached and the answer
    # is a program profile. Coordinator routing already relies on this
    # same deferral for single-word subjects (see the
    # _PROGRAM_SUBJECT_SIGNAL_PATTERN comment above); this extends it to
    # every department-naming question. memory is deliberately NOT passed
    # to this probe call: it only decides whether to defer, and the real
    # search_department() call in the DEPARTMENT branch below records the
    # entity - probing here must not double-record it.
    if _DEPARTMENT_INTENT_PATTERN.search(question_lower):
        if search_department(question, None) is not None:
            return None

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
    # fails.
    #
    # First, restricted to the LONGEST matching subject (Phase 13F fix) -
    # a shorter, more generic subject can be a substring of a longer,
    # more specific one that also matches (e.g. "science" is a substring
    # of "data science", so both "Honours BSc Science" and "Honours BSc
    # Data Science" used to match "What is the Data Science minor?"
    # equally) - confirmed live: without this restriction, the *shorter*
    # match ("Honours BSc Science") could still win purely because it
    # happened to have course-requirement data and the more specific
    # match didn't, which is backwards - specificity has to be decided
    # before that tie-break, not after it. This mirrors the longest-
    # name-first precedent _match_program_name() already uses elsewhere
    # in this file, applied to subject length instead of full name
    # length.
    #
    # A bare subject name is often shared by several program-type
    # variants of the same subject (major, minor, concentration,
    # combined...), so whatever remains tied at that best specificity
    # level is then checked in three passes: "major"-type rows that also
    # have real, structured course-requirement data first (Sprint 11C -
    # among several same-subject major variants, e.g. an Honours BA and
    # an Honours BSc version of Computer Science, prefer whichever one
    # can actually answer a course-requirement follow-up, so
    # establishing a program via a bare subject name and then asking
    # "what do I take in first year?" resolves to a program with real
    # data rather than an arbitrarily-ordered sibling that has none);
    # "major"-type rows generally next, so a bare subject name still
    # resolves to a plain major by default (per Sprint 11A's
    # investigation recommendation) even when none of the candidates
    # have course-requirement data; any other type only as a last-resort
    # fallback.
    candidate_rows = []
    best_subject_length = 0

    for row in rows:

        subject = _strip_to_subject(row[0].lower())

        if not (
            len(subject) >= 3
            and subject in normalized_question
            and _subject_match_is_safe(subject, question_lower)
            and not _subject_match_degree_conflicts(row[0], question_lower)
        ):
            continue

        if len(subject) > best_subject_length:
            best_subject_length = len(subject)
            candidate_rows = [row]

        elif len(subject) == best_subject_length:
            candidate_rows.append(row)

    requirement_program_names = None

    for row in candidate_rows:

        program_type = row["program_type"] if "program_type" in row.keys() else None

        if program_type != "major":
            continue

        if requirement_program_names is None:
            requirement_program_names = set(_all_requirement_program_names())

        if row[0] not in requirement_program_names:
            continue

        if memory is not None:
            memory["last_program"] = row[0]
            _record_entity(memory, "program", row[0], row[0], "search_program")

        return row

    for preferred_types in (("major",), None):

        for row in candidate_rows:

            program_type = row["program_type"] if "program_type" in row.keys() else None

            if preferred_types and program_type not in preferred_types:
                continue

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


# QA Fix Sprint: graduate mirror of the undergraduate list just above -
# confirmed live, "What graduate programs does WLU offer?" had no
# equivalent and fell through to hybrid_search's embedding search,
# which landed on an unrelated "Faculty of Science" marketing page
# instead of an actual program listing. Same intent-gating shape, same
# deterministic programs.db query (level = 'graduate' instead of
# 'undergraduate'), same truncation convention - deliberately identical
# treatment, not a redesign (17 graduate programs total today, well
# under the 30-item cap, so truncation_note is inert but kept for
# consistency/future-proofing).
def _has_graduate_program_list_intent(question_lower):

    return bool(
        "graduate" in question_lower
        and "undergraduate" not in question_lower
        and re.search(r"\bprograms?\b", question_lower)
        and re.search(r"\b(?:available|offered|exist|list|what)\b", question_lower)
    )


def search_graduate_program_list(question, memory=None):

    if not _has_graduate_program_list_intent(question.lower()):
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM programs "
        "WHERE level = 'graduate' ORDER BY program_name"
    )

    names = [row[0] for row in cursor.fetchall()]

    conn.close()

    if not names:
        return None

    total = len(names)
    displayed = names[:30]

    lines = "\n".join(f"- {name}" for name in displayed)

    truncation_note = (
        f"\n(Showing {len(displayed)} of {total} graduate programs. "
        f"Ask about a specific program by name for more detail.)"
        if total > len(displayed) else ""
    )

    return (
        f"Graduate programs at Wilfrid Laurier University "
        f"include:\n{lines}\n{truncation_note}",
        None
    )


# --- Fact-lookup intent: a reusable pattern, not a coordinator-specific
# exception ---
#
# A "fact lookup" question asks for exactly one already-known field on
# an entity (coordinator, email, phone, office) rather than a general
# description of it. Root cause this exists to fix: every structured
# branch below used to build an entity's FULL context (description,
# admission/program requirements, biography, ...) unconditionally, and
# only ever *appended* a fact like the coordinator on top of that - so
# "who is the coordinator for X" returned the entire program/department
# page with the answer buried at the bottom (confirmed live: a 6,630-
# character response for a one-line question). Detecting fact intent
# BEFORE building any context, and returning a small "<Entity>: <name>
# \n<Fact Label>: <value>" context INSTEAD of the full one, fixes this
# for every entity type through one shared mechanism - a new fact key
# only ever needs one new trigger pattern plus one entry in the calling
# branch's `valid_facts` tuple, never a new bespoke code path.
_FACT_TRIGGER_PATTERNS = {
    "coordinator": re.compile(
        r"\b(?:coordinat\w*|advisor|advising|chair|director)\b", re.IGNORECASE
    ),
    "email": re.compile(r"\bemail\b", re.IGNORECASE),
    "phone": re.compile(r"\bphone(?:\s*number)?\b", re.IGNORECASE),
    "office": re.compile(r"\boffice(?:\s*location)?\b", re.IGNORECASE),
}


def _detect_fact_intent(question_lower, valid_facts):
    """The first fact key from `valid_facts` (an ordered sequence of
    keys into _FACT_TRIGGER_PATTERNS) whose trigger pattern matches the
    question, or None. `valid_facts` scopes detection to whichever facts
    are actually meaningful for the calling branch's entity type - e.g.
    "email" is never checked for a program/department question, since
    neither has an email field."""

    for fact_key in valid_facts:

        if _FACT_TRIGGER_PATTERNS[fact_key].search(question_lower):
            return fact_key

    return None


def _fact_context(entity_label, entity_name, fact_label, fact_value, contact_email=None):
    """The shared, minimal context every fact-lookup branch returns:
    just the entity's identifying name and the single requested fact -
    never the entity's full description/biography/requirements.
    `fact_label` is expected to already be entity-prefixed ("Program
    Coordinator", "Department Coordinator") so the graceful-fallback
    text it produces ("Program Coordinator information is not
    available.") matches the exact wording every existing caller/test
    already relies on.

    `contact_email` (production polish): when the requested fact itself
    is missing, but the caller already found a REAL, already-scraped
    @wlu.ca contact address relevant to this exact entity (never
    invented - see _extract_contact_email() below, which only ever
    returns an address that was literally present in already-retrieved
    WLU text), it's appended as a concrete next step instead of leaving
    the decline as a dead end. Ignored whenever fact_value is actually
    present, so an answered fact is never followed by an unrelated
    contact suggestion."""

    if fact_value and fact_value.strip():
        value_text = fact_value.strip()
    else:
        value_text = f"{fact_label} information is not available."
        if contact_email:
            value_text += f" For the most current information, contact {contact_email}."

    return f"{entity_label}: {entity_name}\n{fact_label}: {value_text}"


# Production polish: graceful declines point to a specific, already-
# scraped WLU contact address instead of a generic "consult official
# WLU resources" whenever one genuinely exists for the entity in
# question - never a fabricated/guessed address. Scoped deliberately
# narrow: only the three fact-lookup sites below (program/department
# coordinator, faculty phone/office) call this, since those are the
# only places a specific entity has already been matched AND its own
# already-retrieved text is right there to search - a blind "not
# found" (no course/program/faculty ever matched at all) or the vector
# search's low-confidence gate have no specific entity to draw a
# contact from, so they're deliberately left untouched, generic exactly
# as before.
_CONTACT_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@wlu\.ca", re.IGNORECASE)


def _extract_contact_email(*texts):
    """The first @wlu.ca address literally present in any of `texts`
    (already-retrieved description/admission/coordinator/programs text
    for the SAME entity the user asked about), or None. Never
    constructs or guesses an address - only surfaces one that was
    already scraped verbatim, so a missing fact still never risks
    citing a contact that doesn't actually appear anywhere in the WLU
    data."""

    for text in texts:

        if not text:
            continue

        match = _CONTACT_EMAIL_PATTERN.search(text)

        if match:
            return match.group()

    return None


def _get_department_contact_email(program_source_url):
    """Same d=<id> join _get_department_coordinator() uses to find a
    program's owning department row, but scans that department's own
    description/programs/coordinator text for a contact address instead
    of its coordinator name specifically - a reasonable fallback when
    the program's OWN text has no email of its own, since the joined
    department page frequently carries a general advising/office
    contact even when no specific coordinator is named."""

    department_id = _extract_department_id(program_source_url)

    if not department_id:
        return None

    conn = sqlite3.connect("data/departments.db")
    cursor = conn.cursor()

    cursor.execute("SELECT coordinator, description, programs, source_url FROM departments")

    rows = cursor.fetchall()

    conn.close()

    for coordinator, description, programs, dept_source_url in rows:

        if _extract_department_id(dept_source_url) == department_id:
            return _extract_contact_email(coordinator, description, programs)

    return None


# Signals the user wants the program's *coordinator* specifically, not
# just general program information - e.g. "Who is the program coordinator
# for the Master of Applied Computing?", "Who coordinates the MBA?".
# Delegates to the fact-intent trigger table above (which also
# recognizes "advisor"/"chair"/"director" as the same intent) so the two
# faculty-list/department-list guard call sites elsewhere in this
# cascade and the two coordinator-answering branches below always agree
# on what counts as coordinator intent, rather than drifting apart.
def _has_coordinator_intent(question_lower):

    return _detect_fact_intent(question_lower, ("coordinator",)) is not None


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
# "coordinat*"/"advisor"/"chair"/"director" (Sprint 10E, extended
# alongside the fact-lookup pattern above) are included here too: "who
# is the coordinator of Biology?" has no other academic-signal word at
# all, but asking about an academic coordinator (by any of these
# equivalent titles) is itself never a coincidental, non-WLU usage the
# way "history"/"music"/"english" commonly are.
_DEPARTMENT_ACADEMIC_SIGNAL_PATTERN = re.compile(
    r"\b(?:department|faculty\s+of|program|major|minor|degree|"
    r"coordinat\w*|advisor|advising|chair|director|"
    r"at\s+laurier|at\s+wlu|at\s+wilfrid\s+laurier)\b",
    re.IGNORECASE
)


def _department_name_matches(department_name, question_lower):

    name = department_name.lower()

    # Whole-word/whole-phrase match, not substring containment - this is
    # what stops a name like "Art" from matching inside an unrelated word,
    # on top of the academic-signal gate below.
    #
    # The trailing "\b" is emitted only when the name ends in a word
    # character. A name ending in a non-word character (parentheses:
    # "Management Option (LSBE)", "Geography (GG/ES)") can never match
    # with a trailing boundary - the character after ")" is a space or
    # end-of-text, and neither forms a word boundary, so the whole-phrase
    # match silently fails and those departments become unresolvable by
    # name (same class of bug as the trailing-"\b" note in
    # _strip_person_titles). Confirmed live: DEPT_014 "Management Option
    # (LSBE)" and the exact "Criminology Minor (Faculty of Human and
    # Social Sciences)" row both failed to match for this reason.
    if re.search(r"\w$", name):
        name_pattern = rf"\b{re.escape(name)}\b"
    else:
        name_pattern = rf"\b{re.escape(name)}"

    if not re.search(name_pattern, question_lower):

        # Sprint 2A (BUG4) - trailing-parenthetical base fallback. Stored
        # names like "Computer Science (CP/PC Dept)" / "Geography (GG/ES)"
        # append a code/campus parenthetical users never type - "Tell me
        # about the Computer Science department" names only the base, so
        # the whole-phrase check above fails and the department is
        # unresolvable by name (it used to fall through to a program
        # match instead). When the question explicitly names a department
        # ("department"/"dept"), accept a whole-phrase match on the
        # parenthetical's base. Gated on the department-naming word so a
        # bare topic word ("tell me about geography") is never hijacked
        # into a department match it didn't ask for.
        if _DEPARTMENT_INTENT_PATTERN.search(question_lower):

            paren = re.search(r"\s*\([^)]*\)\s*$", name)

            if paren:
                base = name[:paren.start()].strip()
                if len(base) >= 3 and re.search(
                    rf"\b{re.escape(base)}\b", question_lower
                ):
                    return True

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
        coordinator,
        faculty_name
    FROM departments
    """)

    rows = cursor.fetchall()

    conn.close()

    question_lower = question.lower()

    # Most-specific match wins, not first row in DB order (Sprint 2).
    # Distinct department rows can share a whole-word prefix - "Criminology"
    # and "Criminology Minor (Faculty of Human and Social Sciences)" are
    # separate rows, and several departments also carry duplicate rows for
    # different academic-calendar versions. A row only enters consideration
    # when its FULL name appears in the question, so the longest matching
    # name is the most specific referent the user named; returning the
    # first row instead would resolve "Criminology Minor (Faculty of Human
    # and Social Sciences) department" to the broader "Criminology" page.
    best_row = None
    best_name_len = -1

    for row in rows:

        if _department_name_matches(row[0], question_lower):

            if len(row[0]) > best_name_len:
                best_row = row
                best_name_len = len(row[0])

    if best_row is not None:

        if memory is not None:
            memory["last_department"] = best_row[0]
            _record_entity(memory, "department", best_row[0], best_row[0], "search_department")

        return best_row

    return None


def _strip_person_titles(text):

    text = text.lower()

    # A trailing "\b" after an escaped period never matches (a period
    # followed by a space has no word boundary on either side), which
    # left a stray "." behind and broke the substring match this function
    # exists to enable. The "X." alternative has no trailing boundary
    # requirement - the literal period is itself an unambiguous
    # delimiter.
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

    # Same missing-citation bug course_instructors had (search_faculty_
    # courses_taught()) - source_url is already fetched above for every
    # row, just never carried back to the caller, which returned None as
    # the citation source regardless.
    source_urls = [source_url for name, title, source_url in rows]

    return matched_segment, display_rows, source_urls


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

    # Same missing-citation bug course_instructors had (search_faculty_
    # courses_taught()) - source_url is already fetched above for every
    # row, just never carried back to the caller, which returned None as
    # the citation source regardless.
    source_urls = [source_url for name, title, source_url in rows]

    return matched_segment, display_rows, source_urls


# Maximum Levenshtein (insert/delete/substitute) distance for a question
# word to count as a typo of a stored first/last name, e.g.
# "Mahmod"/"Mahmood" or "Gose"/"Ghose" (both distance 1). Deliberately an
# absolute edit-distance cutoff rather than a similarity ratio: a ratio
# threshold (e.g. rapidfuzz.fuzz.ratio >= 85) also scores unrelated
# same-prefix words as near-matches - "weather" vs the surname
# "Weatherby" scores 87.5 despite being a real, reproduced false
# positive (an off-topic "What's the weather like today?" question
# routed to a faculty profile). Edit distance <=1 still catches every
# genuine single-character typo above while rejecting that case (its
# distance is 2). Only ever consulted as a last resort, after every
# exact tier below has already found nothing.
_MAX_NAME_TYPO_DISTANCE = 1

# Below this length a name token is either too short to reliably
# distinguish from an ordinary English word (whole-word tiers) or too
# short for edit-distance similarity to be meaningful (the fuzzy tier) -
# a 1-2 character typo in a 3-letter name changes its meaning entirely.
_MIN_NAME_TOKEN_LENGTH = 4


def _person_name_tokens(name):

    # Same normalization used everywhere else a stored faculty name is
    # compared against question text: drop trailing credentials ("
    # PhD"), then title words, then lowercase - so "Dr. Jane A. Doe,
    # PhD" and a bare question mention of "Jane" or "Doe" line up.
    return _strip_person_titles(_strip_credentials(name).lower()).split()


class _AmbiguousFacultyMatch:
    """Sentinel returned by search_faculty() when a name fragment (e.g. a
    shared surname) matches more than one distinct faculty member - the
    caller must ask the user to clarify rather than silently picking
    whichever row happened to come back first from the DB."""

    __slots__ = ("candidates",)

    def __init__(self, candidates):
        self.candidates = candidates


def _dedupe_faculty_rows(rows):

    # source_url (row[9]) is the stable per-person key used everywhere
    # else in this file (memory, entity history) - deduping on it here
    # means a person who matched more than one tier's own internal check
    # is never mistaken for two different "candidates".
    unique = {}

    for row in rows:
        unique.setdefault(row[9], row)

    return list(unique.values())


def _collect_full_name_matches(rows, normalized_question):

    matches = []

    for row in rows:

        normalized_name = _strip_person_titles(_strip_credentials(row[0]).lower())

        if len(normalized_name) >= _MIN_NAME_TOKEN_LENGTH and normalized_name in normalized_question:
            matches.append(row)

    return matches


def _collect_first_and_last_matches(rows, question_lower):

    matches = []

    for row in rows:

        name_parts = _person_name_tokens(row[0])

        if len(name_parts) < 2:
            continue

        first_name, last_name = name_parts[0], name_parts[-1]

        if len(first_name) < 3 or len(last_name) < 3:
            continue

        first_pattern = rf"\b{re.escape(first_name)}\b"
        last_pattern = rf"\b{re.escape(last_name)}\b"

        if re.search(first_pattern, question_lower) and re.search(last_pattern, question_lower):
            matches.append(row)

    return matches


def _collect_single_name_matches(rows, question_lower, token_index):

    # token_index=0 -> first name only, token_index=-1 -> last name only.
    # No title word is required (unlike the old surname-only tier) so a
    # bare "Mahmood" or "Ranaweera" - with no "Professor"/"Dr." anywhere
    # in the question - still resolves; the disambiguation step below is
    # what keeps a shared/common name from being silently guessed.
    matches = []

    for row in rows:

        name_parts = _person_name_tokens(row[0])

        if not name_parts:
            continue

        name_token = name_parts[token_index]

        if len(name_token) < _MIN_NAME_TOKEN_LENGTH:
            continue

        if re.search(rf"\b{re.escape(name_token)}\b", question_lower):
            matches.append(row)

    return matches


# Conversational scaffolding stripped before judging whether what's left
# of the question is short enough to treat as a bare name attempt (any
# casing accepted) for fuzzy matching - the same filler-stripping idea
# already used for department-list residual matching
# (_DEPARTMENT_LIST_FILLER_WORDS/_department_list_residual above), here
# applied to person-lookup phrasing specifically. Stripping these first
# (rather than relying on capitalization or word count alone) is what
# lets "Who is mahmoodd?" and "Tell me about ranawera" resolve via typo
# tolerance without capitalization - and is also what keeps an auxiliary
# verb like "have" out of consideration entirely, rather than accidentally
# surviving into a fuzzy comparison and colliding with a real name at
# edit-distance 1 ("have" -> "Dave" - a reproduced false positive).
_NAME_QUERY_FILLER_WORDS = _DEPARTMENT_LIST_FILLER_WORDS | {
    "does", "did", "do", "has", "have", "having", "can", "could", "would",
    "will", "shall", "should", "it", "its", "this", "that", "these",
    "those", "you", "your", "yours", "he", "she", "him", "her", "his",
    "hers", "they", "them", "their", "a", "an", "and", "or", "to",
    "taught", "teaching", "know", "name", "person",
    # Ordinal follow-up vocabulary (_ORDINAL_PATTERN/_ORDINAL_POSITIONS
    # below) - "the first one" is a reserved phrase resolved
    # deterministically by entity-history ordinal resolution, tried only
    # after structured_search (and thus this function) already returns
    # None. Without these stripped, "first" - a common English word -
    # collides with the real surname "Kirst" at edit-distance 1, a
    # reproduced false positive that made search_faculty wrongly resolve
    # before ordinal resolution ever got a chance to run.
    "first", "second", "third", "fourth", "fifth", "last", "one",
    # "more" is common conversational-followup vocabulary (see
    # FOLLOWUP_PHRASES: "more", "more details", "tell me more") with no
    # person-identifying signal of its own, but at edit-distance 1 from
    # the real surname "Moore" - a reproduced false positive: "Tell me
    # more about them" reduced to the single residual word "more" (every
    # other word already filtered), which fuzzy-matched an unrelated
    # faculty member (James Moore) and let search_faculty() resolve the
    # RAW query before resolve_contextual_reference() ever got a chance
    # to correctly resolve "them" against entity_history.
    "more",
}

# At most a first+last name's worth of actual content - short enough
# that the residual (question words minus filler) is almost certainly
# the name attempt itself, so any casing is accepted without requiring
# any other person-intent signal. Deliberately capped at 2, not looser:
# real-world testing found a 3-word residual is where an ordinary topic
# question can land after filler-stripping too (e.g. "What is the
# tuition for Mars students?" strips to "tuition mars students") - and
# "Mars" is one edit from several real first/last names (Marc/Mark/
# Mary/Marsh), a reproduced false positive that hijacked an
# international-tuition question into a faculty-disambiguation prompt.
# 2 stays wide enough for every required typo case (a bare single name
# or a first+last pair, e.g. "ranawera", "Chatura Ranweera", "Who is
# mahmoodd?" all resolve to a residual of 1-2) without accepting a
# 3-word remainder that's just as likely to be ordinary sentence content.
_BARE_NAME_RESIDUAL_MAX_WORDS = 2

# A longer residual means the question has substantial content beyond
# scaffolding - capitalization alone (an ordinary proper noun signal)
# isn't enough there, since real, ordinary proper nouns unrelated to any
# person also collide at edit-distance 1 by chance ("Mars" above). An
# explicit person-intent word - "who", or a title - is required in
# addition, so this branch only ever fires for a question that's
# actually asking about a person, matching the same title-word signal
# already used by _extract_person_query_name elsewhere in this file.
_PERSON_INTENT_PATTERN = re.compile(r"\b(?:who|professor|prof|dr)\.?\b", re.IGNORECASE)


def _fuzzy_candidate_words(question, question_lower):

    words = re.findall(r"[a-z]+", question_lower)
    residual = [word for word in words if word not in _NAME_QUERY_FILLER_WORDS]

    if residual and len(residual) <= _BARE_NAME_RESIDUAL_MAX_WORDS:
        candidates = residual
    elif _PERSON_INTENT_PATTERN.search(question_lower):
        # Skip index 0: the sentence-initial word is capitalized by
        # convention regardless of whether it's a proper noun, so it
        # carries no signal here.
        candidates = [
            word.lower()
            for index, word in enumerate(re.findall(r"[A-Za-z']+", question))
            if index > 0 and word[:1].isupper()
        ]
    else:
        candidates = []

    candidates = [word for word in candidates if len(word) >= _MIN_NAME_TOKEN_LENGTH]

    # A candidate word immediately glued to digits in the original text
    # (no space in between - "GHOST101", "CP312") is an attempted
    # course-code-shaped token, never a person's name: real names are
    # never written glued directly to a number. Without this, a fake/
    # nonexistent course code's letter-prefix can still fuzzy-match a
    # real surname at edit-distance 1 (confirmed live: "Who's the
    # instructor for GHOST101?" fuzzy-matched "Ghose") and produce a
    # confident, wrong faculty profile instead of the graceful decline
    # this typo-tolerance path is supposed to reserve for genuine name
    # typos. Applied after both branches above, so it covers either one
    # uniformly regardless of which produced the candidate.
    return [
        word for word in candidates
        if not re.search(rf"{re.escape(word)}\d", question_lower)
    ]


def _collect_fuzzy_name_matches(rows, question, question_lower):

    # Last-resort typo tolerance: only reached once every exact tier
    # above has matched nothing at all. Compares each candidate word in
    # the question against every faculty member's first and last name
    # using edit-distance, so a small misspelling like "Ranaweer" or
    # "Gose" still resolves instead of falling through to semantic/
    # vector search.
    question_words = _fuzzy_candidate_words(question, question_lower)

    if not question_words:
        return []

    # A single stray word (e.g. "France" in "What is the capital of
    # France?") can fuzzy-match some faculty member's name purely by
    # chance and bypass the domain gate with a false positive.
    # Whenever there's more than one candidate word to check, require
    # at least 2 of them to independently match name tokens of the SAME
    # row - a coincidental one-word collision is no longer enough on
    # its own. A lone candidate word (the documented single-name-typo
    # case, e.g. "ranawera") still only needs to match once, since
    # there's nothing else in the question to corroborate it with.
    required_matches = 1 if len(question_words) == 1 else 2

    matches = []

    for row in rows:

        name_parts = _person_name_tokens(row[0])

        name_tokens = {
            token for token in (name_parts[0], name_parts[-1])
            if name_parts and len(token) >= _MIN_NAME_TOKEN_LENGTH
        } if name_parts else set()

        if not name_tokens:
            continue

        matched_word_count = sum(
            1 for word in question_words
            if any(
                Levenshtein.distance(
                    name_token, word, score_cutoff=_MAX_NAME_TYPO_DISTANCE
                ) <= _MAX_NAME_TYPO_DISTANCE
                for name_token in name_tokens
            )
        )

        if matched_word_count >= required_matches:
            matches.append(row)

    return matches


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
    normalized_question = _strip_person_titles(question_lower)

    # Tiers are tried in order of specificity/confidence; the first tier
    # that matches anything at all wins outright (no falling through to a
    # looser tier just because a stricter one only found one candidate).
    candidates = []

    for tier_matches in (
        # Tier 1: title-stripped full-name substring match.
        _collect_full_name_matches(rows, normalized_question),
        # Tier 2: first name AND last name both present as whole words,
        # in any order (e.g. "Tell me about Louise Dawe" - no title, and
        # the stored "Dr. Louise N. Dawe" won't substring-match because
        # of the middle initial).
        _collect_first_and_last_matches(rows, question_lower),
        # Tier 3: last name only (e.g. "Mahmood", "Ranaweera").
        _collect_single_name_matches(rows, question_lower, -1),
        # Tier 4: first name only.
        _collect_single_name_matches(rows, question_lower, 0),
    ):
        if tier_matches:
            candidates = tier_matches
            break

    # Tier 5: fuzzy/typo tolerance - only consulted once nothing above
    # matched at all.
    if not candidates:
        candidates = _collect_fuzzy_name_matches(rows, question, question_lower)

    if not candidates:
        return None

    candidates = _dedupe_faculty_rows(candidates)

    if len(candidates) > 1:
        return _AmbiguousFacultyMatch(candidates)

    row = candidates[0]

    if memory is not None:
        memory["last_faculty"] = row[0]
        _record_entity(memory, "faculty", row[9], row[0], "search_faculty")
        if row[3]:
            _record_entity(memory, "department", row[3], row[3], "search_faculty")

    return row


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
    # Sprint 2A (BUG3): "Which professors work in Artificial
    # Intelligence?" - the professor/faculty + work/teach/research
    # phrasing names a research TOPIC, not a course (see the course-defer
    # guard at the COURSE branch of structured_search). These run AFTER
    # the plain "who researches X" pattern above so that phrasing keeps
    # its exact current behavior, and the embedding distance threshold
    # downstream is the real gate - a topic with no faculty match simply
    # returns None and the query falls through as before.
    re.compile(
        r"(?:which|what)?\s*professors?\s+(?:who\s+)?"
        r"(?:work|teach|research|specialize)\s+(?:in|on)\s+(.+)",
        re.IGNORECASE
    ),
    re.compile(
        r"faculty\s+(?:who\s+)?(?:work|research|specialize)\s+(?:in|on)\s+(.+)",
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

# Sprint 2A (BUG3) - course-defer guard. "Which professors work in
# Artificial Intelligence?" asks about PEOPLE in a research area, but the
# course branch of structured_search matches the topic word against the
# course name ("Artificial Intelligence" = CP468) and answers with the
# course before the RESEARCH TOPIC aggregation is ever reached. When this
# narrow phrasing fires (an explicit professors/faculty + work/teach/
# research + in/on pattern), the COURSE branch is skipped so execution
# reaches search_faculty_by_research_topic() below - which is itself
# self-limiting via the embedding distance threshold (no faculty match ->
# None -> the query falls through unchanged). Complements the two
# professor/faculty patterns added to _RESEARCH_INTENT_PATTERNS.
_FACULTY_RESEARCH_PHRASING_PATTERN = re.compile(
    r"\b(?:which|what)\s+professors?\s+(?:work|teach|research)\s+(?:in|on)\b|"
    r"\bprofessors?\s+(?:who\s+)?(?:work|teach|research|specialize)\s+(?:in|on)\b|"
    r"\bfaculty\s+(?:who\s+)?(?:work|research|teach|specialize)\s+(?:in|on)\b",
    re.IGNORECASE,
)


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

    # Same missing-citation bug course_instructors had (search_faculty_
    # courses_taught()) - url is already fetched above for every row,
    # just never carried back to the caller, which returned None as the
    # citation source regardless.
    source_urls = [url for url, name, title in rows]

    return topic, display_rows, source_urls


# --- Retrieval quality: metadata-aware reranking ---
#
# search_vector()'s Chroma .query() only ever orders by raw embedding
# distance, which regularly promotes a semantically-nearby-but-wrong
# page (e.g. a news article that happens to mention "scholarship" once)
# over the page whose title/URL is actually about the query's topic -
# and never discounts a chunk that's half navigation/footer boilerplate,
# since chunk.py splits every page into fixed 300-word windows with no
# page-structure awareness at all (a chunk can start mid-nav-list and
# end mid-paragraph). _rerank_vector_candidates() re-scores a WIDER
# candidate pool than what's finally used, adjusting raw distance using
# the two metadata fields actually available on every chunk (title,
# url - see build_vector_db.py) plus literal boilerplate-text
# detection. It never touches the embedding model or the Chroma index
# itself - only the order/selection of what search_vector() already
# retrieved - and it never runs for anything structured_search() already
# answered, so deterministic routing is completely unaffected.

# Retrieve more candidates than will actually be used so reranking has
# real material to promote a better-matching page from beyond the
# original top 5.
_VECTOR_CANDIDATE_POOL_SIZE = 15

# Both patterns are literal, highly-reliable markers of nav/footer text
# that leaked into a content chunk (scrape.py strips <nav>/<footer> tags,
# but "Skip to main content" / "In this section" link lists and the
# cookie-consent banner are rendered outside those tags on wlu.ca, so
# they survive into the scraped body text - see scrape.py/chunk.py).
# Each hit adds a rank penalty (requirement: nav/footer chunks should
# "rank much lower"), rather than trying to surgically cut the matched
# text out of the middle of an otherwise-real content chunk, which risks
# mangling real prose given chunks have no paragraph/sentence boundaries
# preserved to cut cleanly along.
_BOILERPLATE_RANK_PATTERNS = [
    re.compile(r"skip to main content", re.IGNORECASE),
    re.compile(r"we use cookies on this site", re.IGNORECASE),
    re.compile(r"\bin this section\b", re.IGNORECASE),
]
_BOILERPLATE_RANK_PENALTY = 0.08

# A news article is rarely the authoritative page for a general topic
# question ("Scholarships" should prefer a scholarships/financial-aid
# program page over a story about one student's award) - a mild,
# non-dominant penalty so a genuinely on-topic non-news page always wins
# when one exists, without completely hiding a news article that turns
# out to be the ONLY chunk in the corpus actually related to the topic.
_NEWS_URL_PATTERN = re.compile(r"/news/", re.IGNORECASE)
_NEWS_RANK_PENALTY = 0.15

# A page scoped to one specific program ("/programs/<subject>/...") is
# rarely the right source for a broad, university-wide procedural
# question that never names that subject at all - confirmed live:
# "How do I make changes to my course registration?" (a general
# question, no program named) surfaced the Music program's own FAQ
# page as the answer, purely because that FAQ's title happens to
# contain "Course" ("Music Program and Course Offering FAQs") and its
# body happens to mention LORIS in the specific context of music
# ensemble registration - both true, both irrelevant to what was
# actually asked. A different, differently-worded registration
# question ("How do I register for courses?") returned a completely
# different, also-not-quite-right answer (a MyLearningSpace self-
# registration guide - see _MYLS_SELF_REGISTRATION_URL below) for the
# same underlying reason: the corpus was confirmed, by direct search,
# to have no single general "how to register for courses" page at all,
# only scattered incidental mentions across narrow program-specific
# advising pages - so a generic question has no genuinely authoritative
# match to surface. This doesn't fix the missing-source problem - it
# can't invent content that isn't scraped - but applied as a penalty to
# hybrid_rerank.cross_encoder_rerank()'s own scores (see
# _apply_topical_mismatch_penalty() and its call site in
# hybrid_search() below - the cross-encoder, not _rerank_vector_
# candidates() above, is what actually picks the cited page; that
# function's own per-URL dedup is a completely separate, dead code
# path today, not touched here), it stops a narrow program's own FAQ
# from outranking a genuinely general page, or from winning outright
# when nothing general exists, in favor of the LLM's own "the available
# WLU data doesn't contain enough information to answer confidently"
# fallback (already part of generate_answer()'s system prompt, app.py)
# once nothing left outranks it. Only ever fires when the subject truly
# isn't mentioned anywhere in the question - a real "How do I register
# for my Music courses?" question still correctly reaches this same
# page.
_PROGRAM_SPECIFIC_URL_PATTERN = re.compile(r"/programs/([a-z-]+)/", re.IGNORECASE)
_PROGRAM_SPECIFIC_MISMATCH_PENALTY = 6.0

# The MyLearningSpace homepage guide's "Self Registration" section is
# specifically about registering for a short, fixed list of NON-CREDIT
# training modules within the MyLS learning platform (WHIMIS Training,
# Laboratory Safety, Academic Integrity Certificate, Mathematics and
# Statistics Learning Support, Respondus Practice Quiz, Young Worker
# Health and Safety Orientation) - not academic course registration at
# all, which is a genuinely different system (LORIS) and process.
# Confirmed live: "how do i access my mac courses online" (an ambiguous
# acronym - WLU has at least three real "MAC"s: Master of Applied
# Computing, the Milton Academic Centre, and the Mathematics Assistance
# Centre - with no established context to disambiguate it) surfaced
# this page and confidently walked through the self-registration steps
# as if they answered the question, despite having nothing to do with
# any of the three. Penalized whenever the question doesn't mention
# MyLS by name or any of the specific module names this page actually
# covers - a genuine "how do I use MyLS self-registration" question
# still correctly reaches it.
_MYLS_SELF_REGISTRATION_URL = (
    "myls-homepage.html"
)
_MYLS_SELF_REGISTRATION_ON_TOPIC_WORDS = {
    "myls", "mylearningspace", "whimis", "laboratory", "safety",
    "respondus", "worker",
}
_MYLS_SELF_REGISTRATION_MISMATCH_PENALTY = 6.0


def _apply_topical_mismatch_penalty(candidates, question_words):
    """Adjusts hybrid_rerank.cross_encoder_rerank()'s own
    cross_encoder_score downward for a candidate whose URL matches one
    of the confirmed topical-mismatch patterns above, then re-sorts.
    ms-marco-MiniLM-L-6-v2 returns a raw, unbounded relevance logit, not
    a 0-1 probability - confirmed live the winning score for a real
    mismatch case sat at 5.48 with the best remaining (still imperfect,
    but not confidently wrong) alternative at 3.45, a gap of ~2 that a
    smaller penalty was measured to not consistently clear. 6.0 was
    picked empirically against that real gap, not a theoretical
    "handful of points" guess - large enough that a mismatched
    candidate essentially never wins outright unless every other
    candidate in the pool is ALSO penalized, which is exactly the
    "nothing good exists" case this is meant to surface honestly rather
    than paper over.
    Never mutates the candidate dicts hybrid_search() already recorded
    in its debug trace - operates on the same objects in place is fine
    since cross_encoder_rerank() already returns fresh dicts, but the
    re-sort itself is done on a new list so caller code that still holds
    a reference to the pre-penalty order (there isn't any today, but
    this keeps the function honest about not having a hidden side
    effect) is unaffected."""

    for candidate in candidates:

        url = candidate.get("url") or ""

        program_match = _PROGRAM_SPECIFIC_URL_PATTERN.search(url)

        if program_match:

            subject_words = {
                _normalize_rerank_word(w)
                for w in program_match.group(1).split("-")
            }

            if not any(
                _normalize_rerank_word(w) in subject_words
                for w in question_words
            ):
                candidate["cross_encoder_score"] -= (
                    _PROGRAM_SPECIFIC_MISMATCH_PENALTY
                )

        if _MYLS_SELF_REGISTRATION_URL in url:

            if not any(
                _normalize_rerank_word(w) in _MYLS_SELF_REGISTRATION_ON_TOPIC_WORDS
                for w in question_words
            ):
                candidate["cross_encoder_score"] -= (
                    _MYLS_SELF_REGISTRATION_MISMATCH_PENALTY
                )

    return sorted(
        candidates, key=lambda c: c["cross_encoder_score"], reverse=True
    )

# FAQ intent - a question that explicitly asks for "the FAQ(s)" for a
# topic is asking for the topic's FAQ page, not for the topic's
# program/department profile. FAQ content lives only in the scraped
# document corpus (e.g. .../faq.html, .../faqs.html); the structured
# cascade cannot produce it, and its program/department branches would
# instead match the topic itself - confirmed live: "What frequently
# asked questions exist about MSW program requirements?", "What FAQs
# exist about music program and course offerings?" and "What FAQs exist
# for the Social Work professional development offerings?" each returned
# the matching program (or department) profile instead of the topic's
# FAQ page, which hybrid/vector retrieval finds directly. The pattern
# gates BOTH the deferral guard in structured_search() and the FAQ-page
# boost in _apply_faq_intent_boost() below.
_FAQ_INTENT_PATTERN = re.compile(
    r"\bfaqs?\b|\bfrequently\s+asked\s+questions\b",
    re.IGNORECASE,
)

# FAQ intent page boost. The cross-encoder alone is not a reliable judge
# of WHICH page is "the FAQ page" when the question itself asks for one:
# confirmed live (FAQ_003, "What are the FAQs for Sussex LLB
# applicants?"), the actual Sussex LLB FAQ page was the #1 BM25 and #2
# dense candidate, but the program-specific topical-mismatch penalty
# above demoted it by 6.0 (its URL sits under /programs/interdisciplinary/,
# and that penalty reads only the first path segment as the page's
# subject, so "sussex"/"llb" in the question never matched it), while the
# residence-page "Living Learning Program" chunk (which merely happens to
# mention the Laurier-Sussex cluster) won. When the question has explicit
# FAQ intent AND a candidate's URL/title marks it as an FAQ page, the FAQ
# page is unambiguously the requested source, so it gets a boost large
# enough to win outright (same magnitude convention as the penalties
# above). Gated on _FAQ_INTENT_PATTERN, so a non-FAQ question is never
# affected - the inverse case (a GENERAL question that happens to surface
# a program's own FAQ page) is still handled by the program-specific
# penalty above, which fires exactly when the question does NOT name the
# program's subject.
_FAQ_PAGE_INTENT_BOOST = 6.0


def _apply_faq_intent_boost(candidates, question):
    """When `question` has explicit FAQ intent (see _FAQ_INTENT_PATTERN),
    raises cross_encoder_score by _FAQ_PAGE_INTENT_BOOST for every
    candidate whose URL or title marks it as an FAQ page, then re-sorts.
    Same in-place + re-sort contract as _apply_topical_mismatch_penalty();
    a no-op (returns `candidates` untouched) when the question has no FAQ
    intent."""

    if not _FAQ_INTENT_PATTERN.search(question):
        return candidates

    for candidate in candidates:

        url = candidate.get("url") or ""
        title = candidate.get("title") or ""

        if (
            "faq" in url.lower()
            or "faq" in title.lower()
            or "frequently asked" in title.lower()
        ):
            candidate["cross_encoder_score"] += _FAQ_PAGE_INTENT_BOOST

    return sorted(
        candidates, key=lambda c: c["cross_encoder_score"], reverse=True
    )

# Canonical section intent. WLU's service/administrative content
# (academic deadlines & petitions, campus services, student support &
# wellness) is organized into distinct canonical URL sections on
# students.wlu.ca:
#   /calendars-and-petitions/  - dates, deadlines, add/drop, withdraw,
#                                petitions, appeals, graduation, exams
#   /campus-services/          - parking, OneCard, dining, residence,
#                                wifi, safety, study spaces, ...
#   /support-and-wellness/     - international students, mental health,
#                                accessibility, Indigenous, Dean of
#                                Students, counselling, ...
# The cross-encoder reliably RANKS relevant pages but does not know this
# section taxonomy: for many service/administrative questions it picks a
# winning page from the corporate wlu.ca site (governance, discover-
# laurier, strategic-initiatives, future-students) or a sibling
# students.wlu.ca section (finances, academics, campus-services) even
# when the question's canonical section page exists in the corpus -
# confirmed live across 20 Academic Deadlines / Campus Services /
# Student Services benchmark items that all answered correctly but cited
# a page outside their benchmark's canonical section. First match wins;
# a question naming none of these sections is left untouched. Gated on
# _FAQ_INTENT_PATTERN so an FAQ question - already handled by
# _apply_faq_intent_boost() - is never double-preferred.
_SECTION_INTENT_PATTERNS = [
    (re.compile(
        r"\b(deadlines?|due|add\s*/\s*drop|withdraw|petitions?|appeals?|"
        r"appealing|academic\s+calendars?|"
        # QA Fix Sprint: "semester" added alongside "term" (users say
        # both interchangeably; only "term" was covered), plus the
        # reversed word order ("start date for winter term/semester",
        # "beginning of the winter term") - confirmed live, "When does
        # the winter semester start?" and "What is the start date for
        # winter term?" both matched neither the old pattern nor any
        # other structured/canonical-section signal and fell through to
        # a wrong-page vector match (an orientation event page, an
        # unrelated electives page) even though the real dates live on
        # this exact canonical page.
        r"(?:term|semester)\s+(?:starts?|begins?)|"
        r"(?:starts?|begins?|start\s+dates?|beginning)\s+(?:of|for)\s+"
        r"(?:the\s+)?(?:winter|fall|spring|summer)?\s*(?:term|semester)|"
        r"exam-?related|graduation|registration|important\s+dates?|"
        r"last\s+day\s+to\s+drop)",
        re.IGNORECASE,
    ), "/calendars-and-petitions/"),
    (re.compile(
        r"\b(parking|onecard|dining|study\s+spaces?|residenc|"
        r"tech\s+services?|sustainab|classroom|cycling|transit|labs?|"
        r"equipment|special\s+constable|electric\s+vehicle|wifi|wi-fi|"
        r"wireless|accounts?|campus\s+safety|lounge|housing)",
        re.IGNORECASE,
    ), "/campus-services/"),
    (re.compile(
        r"\b(international\s+students?|mental\s+health|wellness|"
        r"accessible\s+learning|indigenous|dean\s+of\s+students|"
        r"athletics|recreation|gendered\s+violence|sexual\s+violence|"
        r"orientation|diversity|equity|immigration|counsell|counseling|"
        r"disabilit|racialized|2slgbtq|advising)",
        re.IGNORECASE,
    ), "/support-and-wellness/"),
    # QA Fix Sprint: "Registrar"/"Student Services" contact questions -
    # confirmed live, both fell through to an unrelated Financial Aid or
    # IT Tech Services page. WLU's real corpus has no page literally
    # named "Registrar" or "Student Services" (its actual institutional
    # name for registration/records/admissions matters is "Service
    # Laurier") - the genuinely relevant, grounded source is the
    # official university contacts directory, which names Service
    # Laurier explicitly ("connect with Service Laurier" for "records
    # and registration") and lists real campus switchboard numbers.
    # Deliberately scoped to CONTACT-intent phrasing only ("contact the
    # registrar", "phone number for student services"), never a bare
    # "student services" mention - that phrase is also legitimately used
    # across many OTHER specific-service pages this must not hijack.
    (re.compile(
        r"\b(?:contact(?:\s+information)?|phone\s+number|email)\b"
        r"[^.?!]{0,20}\b(?:registrar|student\s+services?)\b",
        re.IGNORECASE,
    ), "/about/contacts/"),
]

# Canonical section intent boost. When the question's vocabulary names a
# canonical section (above), a candidate whose URL lives in that section
# is very likely the requested source - the cross-encoder does not know
# WLU's section taxonomy and demonstrably picks wrong-section winners
# (see _SECTION_INTENT_PATTERNS' comment). The magnitude matches the
# penalties' convention and is calibrated live: +5.0, not +3.0, was
# required for DEADLINE_016, whose in-pool important-dates chunks score
# -4.2 to -9.9 after the cross-encoder's boilerplate/nav demotion.
_CANONICAL_SECTION_INTENT_BOOST = 5.0

# When the canonical section's pages are NOT already in the fused pool
# (the dense/BM25 top-k truncated them out), merge the section's top
# BM25-scoring chunks into the pool so the boost has a target. See
# hybrid_search()'s merge block.
_CANONICAL_SECTION_MERGE_TOP_K = 3

# "Dean of Students" must defer to hybrid/vector retrieval (its page
# lives under /support-and-wellness/): structured_search()'s FACULTY
# branch would otherwise catch the capitalized title token "Dean" as a
# faculty member's last name and return professor Jason Dean's profile
# instead of the Dean of Students office page (confirmed live:
# STUDENTSVC_006). Same deferral pattern as the FAQ guard in
# structured_search().
_DEAN_OF_STUDENTS_PATTERN = re.compile(
    r"\bdean\s+of\s+students\b", re.IGNORECASE,
)


def _match_canonical_section(question):
    """Return the canonical-section URL fragment named by `question`'s
    vocabulary (first pattern match wins), or None. FAQ-intent questions
    are excluded here so they keep their own FAQ-page boost path
    (_apply_faq_intent_boost)."""

    if _FAQ_INTENT_PATTERN.search(question):
        return None

    for pattern, section in _SECTION_INTENT_PATTERNS:
        if pattern.search(question):
            return section

    return None


def _apply_canonical_section_preference(reranked, question):
    """When `question` names a canonical section (see
    _match_canonical_section), raises cross_encoder_score by
    _CANONICAL_SECTION_INTENT_BOOST for every candidate whose URL lives
    in that section, then re-sorts. Same in-place + re-sort contract as
    _apply_faq_intent_boost(); a no-op when no section is named or the
    question has FAQ intent."""

    section = _match_canonical_section(question)

    if not section:
        return reranked

    for candidate in reranked:
        if section in (candidate.get("url") or ""):
            candidate["cross_encoder_score"] += _CANONICAL_SECTION_INTENT_BOOST

    return sorted(
        reranked, key=lambda c: c["cross_encoder_score"], reverse=True
    )

# Weighted well above the boilerplate/news penalties above so a genuine
# title/URL topic match always outranks a merely-shorter-distance,
# topically-unrelated page (calibrated live: without this, e.g. "Mars"-
# unrelated candidates could still out-rank a real keyword-matched news
# article once boilerplate+news penalties stacked against it).
_TITLE_URL_MATCH_BONUS = 0.35

# Generic connector/question words carry no topic signal - reused here
# rather than redefined, since the intent is identical to why they're
# filtered for faculty-name matching.
_RERANK_STOPWORDS = _DEPARTMENT_LIST_FILLER_WORDS | {
    "what", "which", "where", "when", "why", "how", "does", "do", "is",
    "are", "tell", "me", "please", "information", "i", "you", "your",
}

# Ubiquitous WLU site-structure words - "academic"/"academics" alone sits
# in roughly half the URL tree (everything under /academics/...),
# "program(s)" and "faculty/faculties" are similarly broad, and every
# single title ends "| Wilfrid Laurier University" - a match against any
# of these carries essentially no discriminating signal and previously
# caused spurious wins (e.g. "Academic Deadlines" matching almost any
# /academics/... page purely via "academic", regardless of whether it
# had anything to do with deadlines). Excluded from keyword-match credit
# entirely; a real topic word (scholarship, admission, tuition, ...)
# is never in this set.
_GENERIC_URL_WORDS = {
    "academic", "academics", "program", "programs", "faculty", "faculties",
    "university", "laurier", "wilfrid", "index", "html", "future", "www",
    "wlu",
}


def _significant_question_words(question):

    words = re.findall(r"[a-z]+", question.lower())

    return [word for word in words if word not in _RERANK_STOPWORDS and len(word) >= 3]


def _normalize_rerank_word(word):

    # Naive singular/plural folding ("admissions" <-> "admission") - not
    # a full stemmer, but sufficient for the plain English topic words
    # this is matching against.
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _url_title_keyword_set(title, url):

    title_words = re.findall(r"[a-z]+", (title or "").lower())
    url_words = re.findall(r"[a-z]+", (url or "").lower())

    tokens = {_normalize_rerank_word(w) for w in title_words + url_words}

    return tokens - _GENERIC_URL_WORDS


def _title_url_match_count(question_words, title, url):

    keyword_set = _url_title_keyword_set(title, url)

    return sum(1 for word in question_words if _normalize_rerank_word(word) in keyword_set)


def _boilerplate_hit_count(text):

    return sum(1 for pattern in _BOILERPLATE_RANK_PATTERNS if pattern.search(text))


# Only the cookie-consent sentence is stripped from context text shown
# to the LLM - it's a fixed, exact, repeated phrase safe to remove
# verbatim (see scrape.py's comment on why it survives tag-stripping).
# The "Skip to main content ... In this section ..." nav-list prefix is
# deliberately NOT text-stripped here for the same reason noted above
# (_BOILERPLATE_RANK_PATTERNS): it has no reliable end boundary to cut
# along without risking real content, so it's handled by rank penalty
# only, not text surgery.
_COOKIE_BANNER_PATTERN = re.compile(
    r"We use cookies on this site to enhance your experience\. By "
    r"selecting [“\"]Accept[”\"] and continuing to use this "
    r"website, you consent to the use of cookies\. Accept",
    re.IGNORECASE
)


def _strip_known_boilerplate_text(text):

    return re.sub(r"\s+", " ", _COOKIE_BANNER_PATTERN.sub(" ", text)).strip()


def _rerank_vector_candidates(question, results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    question_words = _significant_question_words(question)

    # Keep only the best-scoring chunk per URL - Chroma's wider candidate
    # pool often contains two adjacent chunks of the same page at
    # near-identical distance (chunk.py's fixed-window chunking), which
    # would otherwise consume multiple final slots with the same source.
    best_per_url = {}

    for document, metadata, distance in zip(documents, metadatas, distances):

        url = metadata.get("url")

        score = distance
        score += _boilerplate_hit_count(document) * _BOILERPLATE_RANK_PENALTY

        if _NEWS_URL_PATTERN.search(url or ""):
            score += _NEWS_RANK_PENALTY

        score -= (
            _title_url_match_count(question_words, metadata.get("title"), url)
            * _TITLE_URL_MATCH_BONUS
        )

        candidate = {
            "score": score,
            "distance": distance,
            "document": document,
            "url": url,
            "title": metadata.get("title"),
        }

        if url not in best_per_url or score < best_per_url[url]["score"]:
            best_per_url[url] = candidate

    return sorted(best_per_url.values(), key=lambda c: c["score"])


def search_vector(question):

    embedding = model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=_VECTOR_CANDIDATE_POOL_SIZE
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


def _course_card_response(result):

    context = f"""
Course Code: {result[0]}
Course Name: {result[1]}
Credits: {result[2]}
Department: {result[4]}
{_course_level_line(result)}
Description:
{result[3]}
{_course_metadata_section(result)}"""

    return context, result[5], "course"


def _course_clarify_response(candidates):

    pairs = sorted({(row[0], row[1]) for row in candidates})

    if len(pairs) > 6:
        names_text = (
            ", ".join(code for code, name in pairs[:6])
            + f", and {len(pairs) - 6} more"
        )
    else:
        names_text = ", ".join(f"{code} ({name})" for code, name in pairs)

    return (
        "I'm not sure which course you mean - I found multiple "
        f"matching courses: {names_text}. Could you mention the "
        "course code or department?",
        None,
        "course_clarify"
    )


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


# --- Hallucination-prevention: deterministic "not found" detection ---
#
# structured_search()'s existing tiers (search_course/search_program/
# search_faculty) already return a clean `None` the moment nothing
# matches - but `None` is indistinguishable from "this question doesn't
# even mention an entity of this kind", so the caller has no way to
# tell "CP999 looks like a course code, it just isn't a real one" apart
# from "this question isn't about a course at all". Both used to fall
# through identically to hybrid_search()'s ungated vector fallback,
# which will confidently paraphrase whatever 5 chunks happen to be
# nearest - even if none of them are actually relevant - producing a
# fabricated answer with a real-looking but wrong citation. The checks
# below give the three structured entity types a deterministic "not
# found" message instead, using the same shape/keyword signals already
# used elsewhere in this file (e.g. domain_guard's own course-code
# regex) to recognize "this question names an entity of this kind"
# without touching any of the actual matching logic in search_course/
# search_program/search_faculty themselves.

# Same course-code shape used by search_course() (line ~320) and
# domain_guard.COURSE_CODE_PATTERN - duplicated read-only here (not a
# change to search_course's own matching) purely to distinguish "no
# course code shape present at all" from "a code-shaped token is
# present but search_course found no row for it".
_COURSE_CODE_SHAPE_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")


def _course_code_shape(question):

    match = _COURSE_CODE_SHAPE_PATTERN.search(question.upper())

    return match.group() if match else None


# A title word is a strong, deliberate signal the user means a specific
# named person (e.g. "Professor Batman") - unlike a bare capitalized
# word/phrase, which risks colliding with ordinary proper nouns that
# aren't attempted faculty names at all (place names, course/program
# names, etc.), so this stays conservative rather than pattern-matching
# any Title-Case phrase. The title word itself is matched case-
# insensitively (inline (?i:...) - scoped to just that alternation), but
# the captured name is deliberately NOT: it must be genuinely
# capitalized in the original text. This is what distinguishes "Professor
# Batman" (a new name attempt - the capitalized word right after the
# title) from a contextual reference like "that professor" or "Does
# that professor do AI research?" - lowercase text right after the title
# word fails to match at all, so this correctly returns None and leaves
# the question for resolve_contextual_reference (a separate, existing
# pronoun/type-hint resolver - "that professor" is one of its own
# reserved patterns) rather than misreading trailing sentence content as
# an attempted name.
def _extract_person_query_name(question):

    match = re.search(
        r"\b(?:(?i:professor|prof|dr)\.?)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)",
        question
    )

    return match.group(1).strip() if match else None


# A degree-type prefix directly followed by "of"/"in" (e.g. "Bachelor of
# Space Engineering", "Master in Data Science") is a specific, high-
# precision shape for "the user is naming a specific program" - deliberately
# narrower than just checking for the word "program"/"degree" anywhere in
# the question, which would also fire on generic advice questions ("Which
# program should I choose?") that aren't naming any real program at all.
_PROGRAM_NAME_SHAPE_PATTERN = re.compile(
    r"\b(?:bachelor|master|masters|ph\.?d|doctorate|diploma|certificate)s?"
    r"\s+(?:of|in)\s+[a-z].*",
    re.IGNORECASE
)


def _extract_program_query_phrase(question):

    match = _PROGRAM_NAME_SHAPE_PATTERN.search(question)

    if not match:
        return None

    phrase = match.group().strip().rstrip("?.!")

    return re.sub(r"\s+(?:program|degree)$", "", phrase, flags=re.IGNORECASE).strip()


# --- Broad program-overview questions ---
#
# A genuinely broad, entity-less question about the university's program
# offerings as a whole ("What programs does WLU offer?", "What programs
# does Laurier have?") has no single winning page for hybrid_search's
# embedding search to land on - confirmed live, that exact question
# retrieved an unrelated "Incidental Fees Breakdown" chunk, and "Tell me
# about WLU" retrieved an IT-governance policy chunk, neither anywhere
# near an actual program listing. Answered deterministically here for the
# same hallucination-safety reason as every other structured type:
# programs.db/departments.db already hold the real counts and names, so a
# grounded aggregate needs no LLM guess at what "WLU's programs" even
# means.
#
# Reached only after every specific PROGRAM/DEPARTMENT/FACULTY/RESEARCH
# check above has already failed to match, so this can never preempt a
# real "What is the Computer Science program?" or "What programs does the
# Computer Science department offer?" lookup - those already return
# above. Requires an explicit "offer(s/ed/ing)" verb or an explicit
# wlu/laurier/university mention alongside the "what/which/all/list"
# quantifier, so an unrelated use of the word "programs" ("What programs
# are available for financial aid?") doesn't false-positive into this.
_PROGRAM_OVERVIEW_PATTERN = re.compile(
    r"\b(?:what|which|all(?:\s+the)?|list(?:\s+of)?(?:\s+all)?)\s+programs?\b"
    r"[^.?!]{0,40}"
    r"\b(?:wlu|laurier|(?:the\s+)?university|offer|offers|offered|offering)\b",
    re.IGNORECASE,
)


def _distinct_faculty_names():
    """The distinct faculties/schools in departments.db, de-duplicated.
    Names are normalized (periods/whitespace collapsed, compared
    case-insensitively) purely to collapse a known data quirk -
    departments.db has both "Lyle S. Hallman Faculty of Social Work" and
    "Lyle S Hallman Faculty of Social Work" for the same faculty - into
    one entry instead of two near-identical ones that would read as a
    formatting bug. Display spelling is kept from whichever variant
    sorts first, not rewritten. Shared by every deterministic aggregate
    that lists faculties (_program_overview_context(),
    _university_overview_context())."""

    conn = sqlite3.connect("data/departments.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT faculty_name FROM departments "
        "WHERE faculty_name IS NOT NULL AND faculty_name != ''"
    )
    raw_names = [row[0].strip() for row in cursor.fetchall()]
    conn.close()

    seen_keys = set()
    faculties = []

    for name in sorted(raw_names):

        key = re.sub(r"[.\s]+", " ", name).strip().lower()

        if key in seen_keys:
            continue

        seen_keys.add(key)
        faculties.append(name)

    return faculties


def _program_overview_context():
    """Deterministic university-wide summary: total program count broken
    down by level, and the distinct faculties/schools offering them."""

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT level, COUNT(*) FROM programs GROUP BY level")
    level_counts = dict(cursor.fetchall())
    conn.close()

    faculties = _distinct_faculty_names()

    undergrad = level_counts.get("undergraduate", 0)
    graduate = level_counts.get("graduate", 0)
    total = undergrad + graduate

    faculty_list = "\n".join(f"- {name}" for name in faculties)

    return (
        f"Wilfrid Laurier University offers {total} programs across "
        f"{len(faculties)} faculties and schools:\n\n"
        f"{faculty_list}\n\n"
        f"Programs span {undergrad} undergraduate and {graduate} graduate "
        f"offerings, including majors, minors, options, concentrations, "
        f"and combined programs.\n\n"
        f'Ask about a specific program, department, or faculty for full '
        f'details (e.g. "Tell me about the Computer Science program").'
    )


# --- Broad "what faculties/schools does WLU have" questions ---
#
# QA Fix Sprint: confirmed live, "What faculties are available at WLU?"
# matched no existing structured branch and fell through to
# hybrid_search, which landed on an IT-security-policy page purely
# because it happened to contain the phrase "Faculty Profile Training" -
# nowhere near an actual list of WLU's faculties/schools. Deterministic
# here for the same reason as every other aggregate above; reuses
# _distinct_faculty_names() (already shared by _program_overview_context()/
# _university_overview_context()) rather than a third copy of the same
# de-duplication logic.
#
# Gated on the PLURAL "faculties" specifically - unlike the singular
# "faculty" (which this codebase's corpus/vocabulary uses for teaching
# staff, e.g. "who is on the faculty", "which faculty teaches CP312"),
# "faculties" unambiguously means the academic divisions (Faculty of
# Science, Faculty of Arts, ...) in ordinary English, so this can never
# collide with a person/instructor question.
_FACULTY_LIST_QUERY_PATTERN = re.compile(
    r"\b(?:what|which|all(?:\s+the)?|list(?:\s+of)?(?:\s+all)?)\s+"
    r"faculties\b"
    r"|\bfaculties\s+(?:does|do|is|are)\s+(?:wlu|laurier|(?:the\s+)?"
    r"university)\s+(?:have|offer)\b",
    re.IGNORECASE,
)


def _faculty_list_context():

    faculties = _distinct_faculty_names()

    faculty_list = "\n".join(f"- {name}" for name in faculties)

    return (
        f"Wilfrid Laurier University has {len(faculties)} faculties "
        f"and schools:\n\n{faculty_list}\n\n"
        f'Ask about a specific faculty for the departments and programs '
        f'it offers.'
    )


# --- Broad "about the university" questions ---
#
# A genuinely broad "tell me about this university" question has no
# single winning page for hybrid_search's embedding search to land on -
# confirmed live, that exact wording retrieved an unrelated McGill
# scholarship chunk, and "Tell me about WLU" retrieved an IT-governance
# policy chunk. Answered deterministically here for the same
# hallucination-safety reason as every other structured type: every
# figure below is a live COUNT from the same structured DB tables every
# other structured response type already reads (programs.db/
# departments.db/courses.db/faculty.db), never an estimate, and never
# routed through vector similarity search at all.
#
# Reached only after every specific PROGRAM/DEPARTMENT/FACULTY/COURSE/
# RESEARCH check above (and the narrower _PROGRAM_OVERVIEW_PATTERN check
# just above) has already failed to match, so this can never preempt a
# real "What is the Computer Science program?" or "What programs does
# WLU offer?" lookup.
_UNIVERSITY_OVERVIEW_PATTERN = re.compile(
    r"\b(?:tell me (?:more )?about|what is|what's|know more about)\b"
    r"[^.?!]{0,20}"
    r"\b(?:wlu|laurier|wilfrid laurier(?:\s+university)?|this\s+university|"
    r"the\s+university)\b"
    r"|\babout\s+(?:wlu|laurier|this\s+university|the\s+university)\b",
    re.IGNORECASE,
)

# Guards both this pattern and _PROGRAM_OVERVIEW_PATTERN above against
# their "university"/"the university" wording matching a DIFFERENT
# school's name - confirmed live, "What is the tuition at the
# University of Toronto?" matched "the university" and returned a
# confident WLU overview instead of correctly declining as out-of-
# domain, a real hallucination-class regression caught by the
# benchmark's out-of-domain suite. Deliberately narrower than
# domain_guard's own _mentions_other_institution() - that function also
# flags a bare "this university"/"the university" alone as naming some
# OTHER school (tested live: true for "I want to know more about this
# university", the exact query this feature exists to answer), which
# would have broken the primary case. Only fires for an actual
# "University of X" construction or a specific, named other
# institution, never a generic reference back to "this"/"the"
# university.
_OTHER_UNIVERSITY_PATTERN = re.compile(
    r"\buniversity\s+of\s+(?!wilfrid\b|laurier\b)\w+"
    r"|\b(?:harvard|mit|yale|oxford|cambridge|stanford|princeton|"
    r"mcgill|queen'?s|guelph|western|york|ottawa|windsor|brock|"
    r"carleton|ryerson|toronto\s+metropolitan)\s+university\b",
    re.IGNORECASE,
)

# Real, ingested campus location tokens (courses.db's location_text
# column) - only a name that's actually PRESENT in the ingested data is
# ever shown, so this can never state a campus that isn't grounded in
# real corpus content, and silently reflects the current data if the
# corpus is later re-ingested with different coverage.
_CAMPUS_TOKENS = ("Waterloo", "Brantford", "Toronto", "Kitchener", "Milton")


def _grounded_campuses():

    conn = sqlite3.connect("data/courses.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT location_text FROM courses "
        "WHERE location_text IS NOT NULL AND location_text != ''"
    )
    raw_text = " ".join(row[0] for row in cursor.fetchall()).lower()
    conn.close()

    return [name for name in _CAMPUS_TOKENS if name.lower() in raw_text]


def _university_overview_context():
    """Deterministic, fully grounded university-wide overview. States
    only counts and facts directly computable from the ingested data
    (programs.db/departments.db/courses.db/faculty.db); omits anything
    not actually available (e.g. student population isn't tracked
    anywhere in this corpus) rather than estimating it - the same
    "only show what's actually present" discipline
    _program_overview_context() above already follows."""

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT level, COUNT(*) FROM programs GROUP BY level")
    level_counts = dict(cursor.fetchall())
    conn.close()

    undergrad = level_counts.get("undergraduate", 0)
    graduate = level_counts.get("graduate", 0)
    total_programs = undergrad + graduate

    faculties = _distinct_faculty_names()

    conn = sqlite3.connect("data/departments.db")
    department_count = conn.execute(
        "SELECT COUNT(*) FROM departments"
    ).fetchone()[0]
    conn.close()

    conn = sqlite3.connect("data/faculty.db")
    faculty_member_count = conn.execute(
        "SELECT COUNT(*) FROM faculty"
    ).fetchone()[0]
    conn.close()

    campuses = _grounded_campuses()

    campus_line = (
        f"Campuses: {', '.join(campuses)}\n\n" if campuses else ""
    )

    faculty_list = "\n".join(f"- {name}" for name in faculties)

    return (
        f"Wilfrid Laurier University (WLU) offers {total_programs} "
        f"programs ({undergrad} undergraduate, {graduate} graduate) "
        f"across {len(faculties)} faculties and schools, delivered "
        f"through {department_count} departments and over "
        f"{faculty_member_count} faculty members.\n\n"
        f"{campus_line}"
        f"Faculties and Schools:\n{faculty_list}\n\n"
        f"Faculty across these departments conduct research spanning "
        f'many disciplines - ask about research in a specific area '
        f'(e.g. "Who researches artificial intelligence?") to explore '
        f"further.\n\n"
        f"Ask about a specific program, course, faculty member, or "
        f"department for full details."
    )


# --- Policy index lookup (Phase 2) ---
#
# Gated on the literal word "policy"/"policies" being present, which
# nothing else in this cascade triggers on - so this can never collide
# with or shadow any other structured branch, regardless of where it's
# placed. Only number/title/source_url are stored (policies.db); the
# policy BODY text isn't duplicated here and stays reachable through the
# normal vector-search path like any other page, so a more open-ended
# question ("is there a policy about group assignments?") still falls
# through to hybrid_search() rather than being forced through this
# lightweight index.
_POLICY_KEYWORD_PATTERN = re.compile(r"\bpolic(?:y|ies)\b", re.IGNORECASE)
_POLICY_NUMBER_PATTERN = re.compile(
    r"\bpolic(?:y|ies)\b.{0,20}?\b(\d+(?:\.\d+)?)\b", re.IGNORECASE
)


def _policy_number_shape(question):
    """Distinguishes "no policy-number shape present at all" from "a
    number-shaped token is present right after the word policy/policies
    but search_policy() found no row for it" - the same distinction
    _course_code_shape() draws for course codes, and needed for the
    same reason: search_policy() returning None on its own doesn't say
    which of those two happened, and only the second one is definitive
    enough to answer immediately (see the not-found guard in
    structured_search() below)."""

    match = _POLICY_NUMBER_PATTERN.search(question)

    return match.group(1) if match else None


def _policy_body(url):
    """Reconstruct the full policy body text for a policy page URL from
    the main chunk collection.

    policies.db deliberately stores only (policy_number, policy_title,
    source_url) - the body is not duplicated there, it stays in the
    normal crawl -> scrape -> clean -> chunk -> ChromaDB pipeline (see
    load_policies.py), which is exactly where it is fetched from here.
    Chunks for one page carry sequential integer ids (build_vector_db.py
    assigns a global counter in crawl order), so the body is rebuilt by
    concatenating them in id order. Leading navigation and the trailing
    cookie banner that every scraped page carries are trimmed off.

    Returns "" when the page has no chunks - the caller then falls back
    to the title-only stub, preserving today's behaviour."""

    try:
        result = collection.get(where={"url": url})
    except Exception:
        return ""

    ids = result.get("ids") or []
    docs = result.get("documents") or []

    if not docs:
        return ""

    def _id_key(pair):
        try:
            return int(pair[0])
        except (TypeError, ValueError):
            return float("inf")

    text = "\n".join(doc for _, doc in sorted(zip(ids, docs), key=_id_key))

    # Identical leading navigation on every page ends at the "Print |
    # PDF" marker that sits immediately before the real body.
    lead = text.find("Print | PDF")
    if lead != -1:
        text = text[lead + len("Print | PDF"):]

    # Trailing cookie banner common to every scraped page.
    banner = text.find("We use cookies on this site")
    if banner != -1:
        text = text[:banner]

    return text.strip()


def search_policy(question, memory=None):

    if not POLICIES_DB_READY:
        return None

    if not _POLICY_KEYWORD_PATTERN.search(question):
        return None

    conn = sqlite3.connect("data/policies.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT policy_number, policy_title, source_url FROM policies"
    )
    rows = cursor.fetchall()
    conn.close()

    def _return_policy(row):

        if memory is not None:
            _record_entity(
                memory, "policy", row["policy_number"],
                f"{row['policy_number']} - {row['policy_title']}",
                "search_policy",
            )

        body = _policy_body(row["source_url"])

        if body:
            content = (
                f"Policy {row['policy_number']}: {row['policy_title']}\n\n"
                f"{body}"
            )
        else:
            content = (
                f"Policy {row['policy_number']}: {row['policy_title']}"
            )

        return (
            content,
            row["source_url"],
            "policy"
        )

    number_match = _POLICY_NUMBER_PATTERN.search(question)

    if number_match:

        target_number = number_match.group(1)

        for row in rows:
            if row["policy_number"] == target_number:
                return _return_policy(row)

    question_lower = question.lower()

    for row in rows:

        title = (row["policy_title"] or "").strip()

        if len(title) < 4:
            continue

        if re.search(rf"\b{re.escape(title.lower())}\b", question_lower):
            return _return_policy(row)

    return None


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

        # "policy" (Phase 3) never had a legacy memory-dict slot like
        # the four above - it only ever wrote to entity_history - so
        # this reads that back directly via the same recency-aware
        # helper _resolve_typed_value() uses. Substituted as "policy
        # <number>", not the bare number: search_policy() gates on the
        # literal word "policy"/"policies" being present (so a policy
        # follow-up can never be confused with a course/program/
        # department/faculty one), and many real policy titles (e.g.
        # "12.2 Student Code of Conduct") don't themselves contain that
        # word, so the bare number or display name alone wouldn't
        # reliably re-trigger it.
        elif (policy_entity := _latest_entity_of_type(memory, "policy")):
            question = f"policy {policy_entity['entity_id']}"

    # FAQ INTENT
    # Defer to hybrid/vector retrieval. An explicit "what are the FAQs
    # for X?" question asks for X's FAQ page, which the structured
    # cascade can never produce - its PROGRAM/DEPARTMENT branches would
    # instead match X itself (see _FAQ_INTENT_PATTERN's comment for the
    # confirmed live cases). Checked on the ORIGINAL question (before
    # any follow-up substitution above), so a memory follow-up never
    # loses its remembered entity on this account; only genuinely
    # FAQ-naming questions are deflected. Mirrors the Sprint 2
    # department-intent guard's deferral pattern.
    if _FAQ_INTENT_PATTERN.search(question_lower):
        return None

    # DEAN OF STUDENTS
    # Defer to hybrid/vector retrieval. "Who is the Dean of Students and
    # what do they do?" asks about the Dean of Students ROLE/office, whose
    # page lives under /support-and-wellness/, but search_faculty()'s
    # tiered last-name matching catches the capitalized title token "Dean"
    # as a faculty member's last name and returns professor Jason Dean's
    # profile instead (confirmed live: STUDENTSVC_006). Same deferral
    # pattern as the FAQ guard above; checked on the original question.
    if _DEAN_OF_STUDENTS_PATTERN.search(question_lower):
        return None

    # FACULTY COURSES TAUGHT
    # Must run before the plain COURSE lookup below: "Who has taught
    # CP104?" contains a bare course code that search_course() would
    # otherwise match first, returning the course description instead of
    # answering who taught it. Bare "What is CP104?" has no "who has
    # taught" trigger, so it's unaffected and still reaches search_course
    # exactly as before.

    taught_result = search_faculty_courses_taught(question, memory)

    if taught_result:
        return (*taught_result, "course_instructors")

    # FACULTY COURSES TAUGHT - PERSON + TOPIC
    # Must also run before the plain single-person FACULTY lookup further
    # below: "Has Kaiyu Li taught database courses?" contains a full
    # name that search_faculty()'s own tiered matching would otherwise
    # match directly, returning a generic profile instead of answering
    # the actual question.

    person_topic_result = search_faculty_courses_by_topic(question, memory)

    if person_topic_result:
        return (*person_topic_result, "faculty_topic_courses")

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

    # Sprint 2A (BUG3) - faculty research-topic phrasing. "Which
    # professors work in Artificial Intelligence?" names a research TOPIC,
    # not the course whose name happens to contain it (CP468 "Artificial
    # Intelligence") - without this guard search_course() matches on the
    # topic word and answers with the course, and the RESEARCH TOPIC
    # aggregation below (the professors who actually work in the area) is
    # never reached. Skipping the course branch lets execution continue
    # to search_faculty_by_research_topic() - the pattern is narrow and
    # the embedding distance threshold there is the real gate, so a
    # phrasing with no faculty match falls through unchanged.
    faculty_research_phrasing = _FACULTY_RESEARCH_PHRASING_PATTERN.search(
        question_lower
    )

    if not faculty_research_phrasing:
        result = search_course(question, memory)

        if result:
            return _course_card_response(result)

    # A code-shaped token is present (e.g. "CP999") but search_course()
    # found no matching row - definitive enough to answer immediately
    # rather than letting it fall through to the vector fallback, which
    # has no way to know the code doesn't exist and would otherwise
    # paraphrase whatever unrelated chunks happen to be nearest.
    course_code_shape = _course_code_shape(question)

    if course_code_shape:
        return (
            f"I couldn't find a course with code {course_code_shape} in "
            f"the Wilfrid Laurier University data.",
            None,
            "not_found"
        )

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
        return (*program_list_result, "undergraduate_program_list")

    # GRADUATE PROGRAM LIST (QA Fix Sprint) - same reasoning as the
    # undergraduate list immediately above.
    grad_program_list_result = search_graduate_program_list(question, memory)

    if grad_program_list_result:
        return (*grad_program_list_result, "graduate_program_list")

    # PROGRAM

    result = search_program(question, memory)

    if result:

        # Fact lookup, checked BEFORE any of the program's full context
        # is built: a coordinator question gets a concise "Program: X /
        # Program Coordinator: Y" context and returns immediately,
        # instead of that answer being appended to the end of the
        # entire program page (the root cause fixed here - confirmed
        # live, this used to produce a 6,630-character response to a
        # one-line question).
        program_fact_intent = _detect_fact_intent(question_lower, ("coordinator",))

        if program_fact_intent == "coordinator":

            coordinator = _get_department_coordinator(result[3])

            contact_email = None

            if not coordinator:

                description = result["description"] if "description" in result.keys() else None

                contact_email = (
                    _extract_contact_email(description, result[1], result[2])
                    or _get_department_contact_email(result[3])
                )

            context = _fact_context(
                "Program", result[0], "Program Coordinator", coordinator, contact_email
            )

            return (context, result[3], "coordinator")

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

        return (context, result[3], "program")

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

        matched_faculty, faculty_rows, faculty_source_urls = faculty_level_result

        return (
            _format_faculty_list_context("Faculty", matched_faculty, faculty_rows),
            faculty_source_urls,
            "faculty_list"
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

        matched_department, faculty_rows, faculty_source_urls = dept_list_result

        return (
            _format_faculty_list_context("Department", matched_department, faculty_rows),
            faculty_source_urls,
            "department_faculty_list"
        )

    # DEPARTMENT

    result = search_department(question, memory)

    if result:

        # Fact lookup, checked BEFORE any of the department's full
        # context is built - same reusable mechanism and same reasoning
        # as the PROGRAM branch above: reads the existing `coordinator`
        # column directly, never inferred from the free-text
        # description, and returns immediately with a concise context
        # instead of appending the answer to the full department page.
        department_fact_intent = _detect_fact_intent(question_lower, ("coordinator",))

        if department_fact_intent == "coordinator":

            coordinator = result["coordinator"]

            contact_email = (
                None if coordinator
                else _extract_contact_email(result[2], result[1])
            )

            context = _fact_context(
                "Department", result[0], "Department Coordinator", coordinator, contact_email
            )

            return (context, result[3], "coordinator")

        faculty_name = result["faculty_name"]
        coordinator = result["coordinator"]

        # Production Polish Sprint: the department profile card never
        # showed the coordinator inline - it was only reachable through
        # a separate "who is the coordinator of X" fact-lookup question
        # (the branch just above), even though departments.db already
        # has the data on the very same row. Added here, "only if
        # populated" (same discipline as every other optional field in
        # this context - Faculty/Level above), so a department with no
        # coordinator on file renders exactly as before.
        context = f"""
Department: {result[0]}
{f"Faculty: {faculty_name.strip()}" if faculty_name and faculty_name.strip() else ""}
{_department_level_line(result)}
{f"Coordinator: {coordinator.strip()}" if coordinator and coordinator.strip() else ""}
Programs:
{result[1]}

Description:
{result[2]}
"""

        return (context, result[3], "department_profile")

    # FACULTY

    result = search_faculty(question, memory)

    if isinstance(result, _AmbiguousFacultyMatch):

        names = sorted({row[0] for row in result.candidates})

        return (
            "I'm not sure which professor you mean - I found multiple "
            f"faculty members matching that name: {', '.join(names)}. "
            "Could you provide a full name or more detail (e.g. their "
            "department)?",
            None,
            "faculty_clarify"
        )

    if result:

        # Fact lookup (same reusable mechanism as PROGRAM/DEPARTMENT
        # above): "what's their email/phone/office" gets a concise
        # "Name: X / Email: Y" context instead of the full profile
        # (contact block + biography + research interests). Only
        # reached once search_faculty() has already matched one specific
        # named person, so this only ever narrows an already-resolved
        # question - it can never cause a false person match on its own.
        faculty_fact_intent = _detect_fact_intent(
            question_lower, ("email", "phone", "office")
        )

        if faculty_fact_intent:

            fact_value = {
                "email": result[4], "phone": result[5], "office": result[6],
            }[faculty_fact_intent]

            fact_label = faculty_fact_intent.capitalize()

            # Only offered when the missing fact ISN'T email itself
            # (suggesting "email them at <their own missing email>"
            # would be circular/meaningless) - the person's own email
            # column, already a real address on file, not free text to
            # search.
            contact_email = (
                result[4] if faculty_fact_intent != "email" and not fact_value
                else None
            )

            context = _fact_context(
                "Name", result[0], fact_label, fact_value, contact_email
            )

            return context, result[9], "faculty_profile"

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

        return context, result[9], "faculty_profile"

    # RESEARCH TOPIC
    # Tried last, after every exact/structured lookup above (course,
    # program, coordinator, faculty-level list, department-level list,
    # department, single-person) - this is the least specific signal in
    # the cascade and must never preempt any of those.

    research_topic_result = search_faculty_by_research_topic(question, memory)

    if research_topic_result:

        topic, faculty_rows, faculty_source_urls = research_topic_result

        return (
            _format_faculty_list_context("Research Topic", topic, faculty_rows),
            faculty_source_urls,
            "research"
        )

    # FACULTY - NOT FOUND
    # Reaching this point means search_faculty() above already returned
    # plain None (a real match or an ambiguous-match clarification would
    # have returned already, back at the FACULTY block). Only fires when
    # a capitalized name actually follows a title word - a genuine new
    # name attempt like "Professor Batman" - rather than falling through
    # to the vector fallback and risking a fabricated profile. A
    # contextual reference like "that professor" has no name to extract
    # (see _extract_person_query_name), so it correctly falls through
    # this check untouched and reaches resolve_contextual_reference (in
    # app.py, tried after structured_search) instead.
    attempted_name = _extract_person_query_name(question)

    if attempted_name:

        return (
            f"I couldn't find a faculty member named {attempted_name} in "
            f"the Wilfrid Laurier University data.",
            None,
            "not_found"
        )

    # PROGRAM - NOT FOUND
    # Same reasoning as the course/faculty checks above: reaching here
    # means search_program() already returned None, so a degree-type-
    # shaped phrase ("Bachelor of X", "Master in X") is a strong enough
    # signal that a specific (nonexistent) program was named.
    program_phrase = _extract_program_query_phrase(question)

    if program_phrase:

        return (
            f'I couldn\'t find a program called "{program_phrase}" in '
            f"the Wilfrid Laurier University data.",
            None,
            "not_found"
        )

    # BROAD PROGRAM-OVERVIEW
    # Reached only once every specific PROGRAM/DEPARTMENT/FACULTY/
    # RESEARCH check above has already failed to match - see
    # _PROGRAM_OVERVIEW_PATTERN's own comment. _OTHER_UNIVERSITY_PATTERN
    # guard: see that pattern's own comment (regression fix).
    if (
        _PROGRAM_OVERVIEW_PATTERN.search(question_lower)
        and not _OTHER_UNIVERSITY_PATTERN.search(question_lower)
    ):

        return (_program_overview_context(), None, "program_overview")

    # BROAD FACULTY/SCHOOL LIST (QA Fix Sprint) - checked before the
    # generic university overview so "What faculties are available at
    # WLU?" gets the focused list, not the full university summary. Same
    # _OTHER_UNIVERSITY_PATTERN guard as the checks around it.
    if (
        _FACULTY_LIST_QUERY_PATTERN.search(question_lower)
        and not _OTHER_UNIVERSITY_PATTERN.search(question_lower)
    ):

        return (_faculty_list_context(), None, "faculty_list_overview")

    # BROAD "ABOUT THE UNIVERSITY" OVERVIEW
    # Checked after the narrower program-overview pattern immediately
    # above, so "Tell me about WLU's programs" still resolves to the
    # program-specific summary rather than the more generic one here -
    # see _UNIVERSITY_OVERVIEW_PATTERN's own comment. Same
    # _OTHER_UNIVERSITY_PATTERN guard as immediately above.
    if (
        _UNIVERSITY_OVERVIEW_PATTERN.search(question_lower)
        and not _OTHER_UNIVERSITY_PATTERN.search(question_lower)
    ):

        return (_university_overview_context(), None, "university_overview")

    # POLICY INDEX (Phase 2)
    # Gated on the word "policy"/"policies" (see search_policy's own
    # comment) - safe to try before the course-name fallback below since
    # that gate makes collision with course/program/department/faculty
    # names structurally impossible.
    policy_result = search_policy(question, memory)

    if policy_result:
        return policy_result

    # A number-shaped token is present right after "policy"/"policies"
    # (e.g. "policy 0.0") but search_policy() found no matching row -
    # definitive enough to answer immediately rather than letting it
    # fall through to the vector fallback, which has no way to know the
    # number doesn't exist and would otherwise paraphrase whatever
    # unrelated chunks happen to be nearest - confirmed live, this was
    # exactly the source of non-deterministic pass/fail on fake policy
    # numbers in the benchmark (same failure mode _course_code_shape()
    # already guards against for fake course codes, mirrored here).
    policy_number_shape = _policy_number_shape(question)

    if policy_number_shape:
        return (
            f"I couldn't find a policy numbered {policy_number_shape} in "
            f"the Wilfrid Laurier University data.",
            None,
            "not_found"
        )

    # COURSE NAME (last resort)
    # Tried dead last, after every other structured capability above -
    # course-name matching (a bare substring match against the entire
    # ~4600-row catalog) is the least specific signal in this whole
    # cascade, even less specific than the embedding-based research-topic
    # search above it: plenty of ordinary department/research/program
    # words are ALSO literal course titles ("Marketing", "Economics",
    # "Machine Learning", "Consumer Behaviour" all are real course
    # names) - confirmed live, this preempted "Who works in Marketing?"
    # and "Who researches machine learning?" when tried earlier in the
    # cascade, a regression caught during verification and moved here
    # rather than shipped. Only reached once nothing above already
    # answered the question.
    course_name_result = _search_course_by_name(question, memory)

    if isinstance(course_name_result, _AmbiguousCourseMatch):
        return _course_clarify_response(course_name_result.candidates)

    if course_name_result:
        return _course_card_response(course_name_result)

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
# QA Fix Sprint: "this X" added alongside "that X"/"the X" for all four
# types - confirmed live, "Tell me more about THIS program." (arguably
# the most natural of the three phrasings) matched none of these
# patterns and fell through to the generic reference loop below
# instead, which has no per-type awareness and resolved back to
# whatever entity was most recent regardless of type (the just-
# established COURSE, not the program) - see _derive_parent_program()
# for the other half of this fix.
_TYPE_HINTED_PATTERNS = [
    (re.compile(r"\bthat professor\b", re.IGNORECASE), "faculty"),
    (re.compile(r"\bthe professor\b", re.IGNORECASE), "faculty"),
    (re.compile(r"\bthis professor\b", re.IGNORECASE), "faculty"),
    (re.compile(r"\bthat course\b", re.IGNORECASE), "course"),
    (re.compile(r"\bthe course\b", re.IGNORECASE), "course"),
    (re.compile(r"\bthis course\b", re.IGNORECASE), "course"),
    (re.compile(r"\bthat department\b", re.IGNORECASE), "department"),
    (re.compile(r"\bthe department\b", re.IGNORECASE), "department"),
    (re.compile(r"\bthis department\b", re.IGNORECASE), "department"),
    (re.compile(r"\bthat program\b", re.IGNORECASE), "program"),
    (re.compile(r"\bthe program\b", re.IGNORECASE), "program"),
    (re.compile(r"\bthis program\b", re.IGNORECASE), "program"),
]

# "they"/"them"/"she"/"he"/"his"/"her"/"him" most commonly refer to a
# person in English, so faculty is tried first, falling back to the
# standard priority below only if no faculty is on record.
_PERSON_HINTED_PATTERNS = [
    re.compile(r"\bthey\b", re.IGNORECASE),
    re.compile(r"\bthem\b", re.IGNORECASE),
    re.compile(r"\bshe\b", re.IGNORECASE),
    re.compile(r"\bhe\b", re.IGNORECASE),
    re.compile(r"\bhis\b", re.IGNORECASE),
    re.compile(r"\bher\b", re.IGNORECASE),
    re.compile(r"\bhim\b", re.IGNORECASE),
]

# "this"/"that"/"these"/"those" immediately followed by a word that
# names the INSTITUTION ITSELF (not any of the four tracked sub-entity
# types) is a determiner phrase referring to WLU as a whole - "this
# university", "that campus" - never a genuine reference back to an
# established course/program/department/faculty member. Confirmed
# live: after establishing the Master of Applied Computing program as
# context, "events in this university" incorrectly resolved by
# substituting the established program's name in for "this" (the
# generic fallback substitution below has no way to know "university"
# was the actual referent, not the program), producing a completely
# unrelated program card for a question that was never about that
# program at all - a query with no real reference signal getting
# colored by stale entity context. "this course"/"this program"/etc.
# (a real reference to one of the four tracked types) are deliberately
# NOT excluded here - only the institution-level nouns below are,
# since those are never resolvable against any of the four memory
# slots to begin with.
_INSTITUTION_REFERENCE_EXCLUSION = (
    r"(?!\s+(?:university|school|college|campus|institution)\b)"
)

# Bare, type-agnostic references - tried against each memory slot in the
# same priority order structured_search() already uses for the
# follow-up-phrase mechanism.
_GENERIC_REFERENCE_PATTERNS = [
    re.compile(r"\bit\b", re.IGNORECASE),
    re.compile(r"\bits\b", re.IGNORECASE),
    re.compile(rf"\bthat\b{_INSTITUTION_REFERENCE_EXCLUSION}", re.IGNORECASE),
    re.compile(rf"\bthis\b{_INSTITUTION_REFERENCE_EXCLUSION}", re.IGNORECASE),
    re.compile(rf"\bthose\b{_INSTITUTION_REFERENCE_EXCLUSION}", re.IGNORECASE),
    re.compile(rf"\bthese\b{_INSTITUTION_REFERENCE_EXCLUSION}", re.IGNORECASE),
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

# Cold-start follow-up phrases ("tell me more", "explain", "show me",
# ...) carry no topic of their own - they only mean anything once
# FOLLOWUP MEMORY (structured_search) has an established entity to
# substitute for them. With nothing ever established in the
# conversation, hybrid_search's vector fallback would otherwise answer
# confidently from whatever chunk happens to be nearest by embedding
# distance (confirmed live: a cold-start "tell me more" produced a
# confident answer about an unrelated sexual-violence-response page),
# exactly the fabricated-topic failure the _is_referentless_query gate
# above already blocks for bare "it"/"this"/"that" queries.
_FOLLOWUP_NO_CONTEXT_MESSAGE = (
    "It looks like you're following up on something from earlier in "
    "our conversation, but I don't have any prior context to draw on "
    "yet. Could you ask your full question?"
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

# Sprint 2A (BUG6) - a bare campus-qualifier follow-up ("For Brantford?",
# "At Waterloo?", "For the Milton campus?"). The qualifier alone names no
# entity the type/person/generic reference loops can resolve - it modifies
# the conversation's current TOPIC, which for vector-answered intent turns
# is recorded as a "topic" entity (see hybrid_search). Strictly whole-query
# anchored so a full question that merely ends in a campus name ("Where
# can I study at Waterloo?") is never intercepted as a follow-up.
#
# QA Fix Sprint regression: the "what|how about" prefix used to still
# require a following "for/in/at" (the original single alternation put
# the prefix and the preposition in sequence, both effectively
# mandatory) - confirmed live, "What about Waterloo?" after a Brantford
# convocation question fell all the way through to the off-topic gate
# instead of inheriting the convocation topic, even though "For
# Brantford?" (same shape, different preposition) already worked.
# "what|how about" is a complete, unambiguous qualifier on its own -
# "for/in/at" is now optional after it, never required.
_CAMPUS_QUALIFIER_PATTERN = re.compile(
    r"^(?:"
    r"(?:what|how)\s+about\s+(?:(?:for|in|at)\s+)?"
    r"|and\s+(?:for|in|at)\s+"
    r"|(?:for|in|at)\s+"
    r")(?:the\s+)?(waterloo|brantford|milton)(?:\s+campus)?[.?!]*\s*$",
    re.IGNORECASE,
)

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


def _attempt_coordinator_resolution(memory):
    """Shared by both a pronoun-triggered coordinator reference ("who
    coordinates it?", via _attempt_contextual_resolution() below) and a
    bare, pronoun-less one ("Who is the coordinator?", via
    resolve_contextual_reference()'s own top-level check) - both are the
    exact same question once "coordinat..." has been recognized, and
    should resolve identically regardless of which phrasing triggered
    them."""

    target = _resolve_coordinator_target(memory)

    if not target:
        return ("clarify", _GENERIC_CLARIFICATION_MESSAGE)

    entity_type, value = target

    rewritten_question = _COORDINATOR_REWRITE_TEMPLATES[entity_type].format(value=value)

    result = structured_search(rewritten_question, memory)

    if result:
        context, source, response_type = result
        return ("resolved", context, source, response_type)

    return ("clarify", _CLARIFICATION_MESSAGES[entity_type])


def _attempt_intent_rewrite_resolution(question, memory):
    """Checks each _INTENT_REWRITE_RULES pattern against the raw
    question and, on a match, rewrites it against the relevant memory
    entity and resolves through structured_search() - shared by
    _attempt_contextual_resolution() below (pronoun-triggered, e.g.
    "does it have prerequisites?") and resolve_contextual_reference()'s
    own top-level check (bare, pronoun-less, e.g. "What about the
    prerequisites?") so both phrasings of the same question resolve
    identically. Returns None if no rule matches (caller decides what
    to try next), never a false negative on a genuine match."""

    for rule_pattern, rule_type, template in _INTENT_REWRITE_RULES:

        if not rule_pattern.search(question):
            continue

        value = _resolve_typed_value(memory, rule_type)

        if not value:
            return ("clarify", _CLARIFICATION_MESSAGES[rule_type])

        rewritten_question = template.format(value=value)

        result = structured_search(rewritten_question, memory)

        if result:
            context, source, response_type = result
            return ("resolved", context, source, response_type)

        return ("clarify", _CLARIFICATION_MESSAGES[rule_type])

    return None


def _derive_parent_program(memory):
    """When no program has been established directly but a COURSE has
    (e.g. "What is CP683?" then "Tell me more about this program."),
    looks up the course's containing program via
    program_course_requirements - the same table
    _handle_reverse_program_requirement_lookup() already uses for the
    free-text "what program requires CP683?" question, just triggered
    from the opposite direction (memory-driven, not a re-typed course
    code). Confirmed live: without this, "this program" after CP683
    fell through to the generic reference loop below, which has no
    per-type awareness and resolved back to the COURSE itself (CP683
    again) instead of its parent program (Master of Applied Computing).

    Returns the program name only when exactly one program lists the
    course as a requirement - a course shared by several programs (e.g.
    a common elective) has no single "the" parent program to infer, so
    this deliberately returns None (falls through to the ordinary
    clarification) rather than guessing which one the user means."""

    course_code = _resolve_typed_value(memory, "course")

    if not course_code:
        return None

    code_match = re.match(r"\s*([A-Z]{2,4}\d{3}[A-Z]?)", course_code.upper())

    if not code_match:
        return None

    conn = sqlite3.connect("data/programs.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT program_name FROM program_course_requirements "
        "WHERE course_code = ?",
        (code_match.group(1),)
    )

    programs = [row[0] for row in cursor.fetchall()]

    conn.close()

    return programs[0] if len(programs) == 1 else None


def _attempt_contextual_resolution(question, pattern, type_priority, memory):

    if _COORDINATOR_REWRITE_PATTERN.search(question):
        return _attempt_coordinator_resolution(memory)

    intent_result = _attempt_intent_rewrite_resolution(question, memory)

    if intent_result:
        return intent_result

    for entity_type in type_priority:

        value = _resolve_typed_value(memory, entity_type)

        if not value and entity_type == "program":
            value = _derive_parent_program(memory)

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
            context, source, response_type = result
            return ("resolved", context, source, response_type)

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
            context, source, response_type = result
            return ("resolved", context, source, response_type)

        return ("clarify", _CLARIFICATION_MESSAGES.get(entity_type, _GENERIC_CLARIFICATION_MESSAGE))

    substituted_question = _ORDINAL_PATTERN.sub(value, question, count=1)

    result = structured_search(substituted_question, memory)

    if result:
        context, source, response_type = result
        return ("resolved", context, source, response_type)

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

    # Sprint 2A (BUG6) - campus-qualifier follow-up, checked before the
    # reference loops because the qualifier carries no reference marker
    # for them to trigger on. "For Brantford?" qualifies the current
    # TOPIC (recorded as a "topic" entity when a convocation intent turn
    # is answered through hybrid_search), so it is rewritten to
    # "<topic> <campus>" and answered through hybrid_search - reusing the
    # prior context instead of answering the bare qualifier fresh
    # (confirmed: "For Brantford?" after "When is Fall2026 convocation?"
    # was answered without any memory of the prior question). Falls
    # through unchanged when no topic entity exists or the rewritten
    # query fails the vector gate.
    campus_match = _CAMPUS_QUALIFIER_PATTERN.search(question_lower)

    if campus_match:
        topic_entity = _latest_entity_of_type(memory, "topic")
        if topic_entity is not None:
            rewritten = (
                f"{topic_entity['entity_id']} {campus_match.group(1)}"
            )
            result = hybrid_search(rewritten, memory)
            if result is not None and result[2] != "not_found":
                return ("resolved",) + result

    if _COMPARE_PATTERN.search(question_lower):
        return _compare_clarification(memory)

    ordinal_match = _ORDINAL_PATTERN.search(question_lower)

    if ordinal_match:
        return _attempt_ordinal_resolution(question, ordinal_match.group(1), memory)

    # A bare "coordinator" question ("Who is the coordinator?", "Who's
    # the program coordinator?") carries no pronoun/reference-type
    # marker at all for the loops below to trigger on, unlike "who
    # coordinates it?" (which has "it"). Without this check, such a
    # question never reaches _attempt_contextual_resolution()'s own
    # coordinat... handling and falls straight through this function,
    # eventually reaching domain_guard's off-topic gate - which has no
    # memory of the conversation and, confirmed live, classifies the
    # bare word "coordinator" as off-topic on its own merits, producing
    # a decline instead of correctly resolving against the already-
    # established program/department. Checked unconditionally here,
    # exactly like the pronoun-triggered path already does inside
    # _attempt_contextual_resolution(), since "coordinat..." is itself
    # already a strong, unambiguous signal that needs no pronoun to
    # corroborate it.
    if _COORDINATOR_REWRITE_PATTERN.search(question_lower):
        return _attempt_coordinator_resolution(memory)

    # QA Fix Sprint: same reasoning as the bare-coordinator check just
    # above - a bare "What about the prerequisites?" (no pronoun) never
    # reached _attempt_contextual_resolution() (only ever called once a
    # pronoun/reference-marker pattern below has already matched), so it
    # fell through this whole function with no course context at all.
    # Confirmed live: after "What is CP683?", this returned an unrelated
    # chemistry course's prerequisites instead of CP683's own -
    # hybrid_search() had no idea "the prerequisites" meant CP683's.
    # _INTENT_REWRITE_RULES's own patterns ("prerequisites?", "teach...")
    # are themselves already unambiguous enough to need no pronoun,
    # exactly like "coordinat..." above.
    intent_result = _attempt_intent_rewrite_resolution(question_lower, memory)

    if intent_result:
        return intent_result

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


# Calibrated against real distances returned by the main
# wlu_chatbot_chunks collection (same embedding model,
# all-MiniLM-L6-v2, as the faculty-research collection's own calibrated
# _RESEARCH_TOPIC_DISTANCE_THRESHOLD, whose approach this mirrors):
# genuine in-domain questions answerable only through this fallback
# ("What clubs are available at WLU?", "How much does residence cost?")
# land at or below ~1.19, while fabricated-entity questions ("What is
# the tuition for Mars students?") land at ~1.23+. Gates whether the top
# hit is used at all - the retrieval call itself (search_vector, same
# n_results=5, same ordering) is unchanged.
_VECTOR_SEARCH_DISTANCE_THRESHOLD = 1.2

_NO_CONFIDENT_MATCH_MESSAGE = (
    "I couldn't find reliable information about that in the Wilfrid "
    "Laurier University data. Could you rephrase your question or ask "
    "about something more specific?"
)

# --- Referent-less pronoun/reference queries reaching the vector fallback ---
#
# resolve_contextual_reference() (app.py, tried before hybrid_search) only
# ever intervenes when memory already has established context to resolve
# a reference against (its own _memory_has_any_context guard) - by design,
# so it doesn't misread an ordinary sentence that merely contains "this"/
# "that" as a follow-up when nothing has been established yet. But that
# means a query which IS structurally a bare pronoun/reference query -
# "Does it have prerequisites?", "Can I take it in the fall?" - reaches
# hybrid_search()'s vector fallback below with no safeguard at all,
# regardless of memory state: search_vector() has no way to know "it" is
# undefined and will confidently answer using whatever chunk happens to
# be nearest by embedding distance (confirmed live: "Does it have
# prerequisites?" on a fresh session returned a confident answer about
# CH110; "Can I take it in the fall?" returned one about COOP000 - both
# fabricated relative to what the user actually asked). _is_referentless_
# query() reuses the same _TYPE_HINTED_PATTERNS/_GENERIC_REFERENCE_
# PATTERNS resolve_contextual_reference() already recognizes as reference
# markers, so the two functions never disagree about what counts as one.
_REFERENCE_MARKER_PATTERNS = (
    [pattern for pattern, _ in _TYPE_HINTED_PATTERNS]
    + _GENERIC_REFERENCE_PATTERNS
)

# Basic English scaffolding words that carry no topic-specific meaning of
# their own - used only to judge whether a query that already matches a
# reference marker above has any real topical content BESIDES the
# referent itself. Deliberately broader than _NAME_QUERY_FILLER_WORDS
# (which is tuned for person-name residual detection specifically):
# personal pronouns ("I"), generic weak verbs ("take", "get"), and common
# prepositions are exactly the padding words in "Can I take it in the
# fall?" that add no answerable substance once "it" is removed.
_REFERENTLESS_SCAFFOLDING_WORDS = {
    "i", "me", "my", "we", "us", "our", "you", "your",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "can", "could", "would", "will", "shall", "should", "may", "might", "must",
    "have", "has", "had", "having",
    "take", "takes", "taking", "get", "gets", "getting", "go", "goes", "going",
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "about",
    "and", "or", "so", "please",
}

# A query like "What is the philosophy behind this decision?" still
# leaves several real content words ("philosophy", "behind", "decision")
# after its reference marker ("this") and scaffolding are stripped - that
# must keep working exactly as before. 1 is generous enough to still
# catch both confirmed bug cases (each leaves exactly one leftover word:
# "prerequisites", "fall") while requiring at least 2 genuine content
# words before a query is treated as carrying its own topic.
_REFERENTLESS_CONTENT_WORD_LIMIT = 1


def _is_referentless_query(question):
    """True for a query that's ENTIRELY a bare pronoun/reference marker
    (the same patterns resolve_contextual_reference() recognizes) plus
    generic sentence scaffolding, with no substantive topic content of
    its own. Checked independently of memory state - see this section's
    module-level comment for why that matters."""

    question_lower = question.lower()

    if not any(
        pattern.search(question_lower) for pattern in _REFERENCE_MARKER_PATTERNS
    ):
        return False

    residual_text = question_lower

    for pattern in _REFERENCE_MARKER_PATTERNS:
        residual_text = pattern.sub(" ", residual_text)

    words = re.findall(r"[a-z']+", residual_text)
    content_words = [w for w in words if w not in _REFERENTLESS_SCAFFOLDING_WORDS]

    return len(content_words) <= _REFERENTLESS_CONTENT_WORD_LIMIT


# ------------------------------------------------------------------
# Sprint C - multi-document context construction (retrieval
# orchestration only). hybrid_search() below still selects the exact
# same winning page and cites it as the primary source; this block only
# changes what CONTEXT is handed to the answer generator. The winning
# page's own chunks are kept verbatim (ranked by fused score, exactly
# as before Sprint C), then a small, bounded set of COMPLEMENTARY
# chunks from the best OTHER pages in the same relevance-ranked pool is
# appended - after near-duplicate suppression - so the final context
# contains genuinely different information instead of a single page's
# slice, and never repeated chunks.
#
# Ranking, winner selection, the distance gate, structured search,
# BM25, vector search, and Chroma are untouched. Citations gain the
# secondary sources because citation.build_citation() already accepts
# an iterable of URLs - the same mechanism structured_search()'s
# multi-instructor answer uses (see _format_faculty_list_context near
# line 1652) - and the renderer + benchmark check iterate
# citation["sources"], so no code outside retriever.py changes.
# ------------------------------------------------------------------

_MULTIDOC_MAX_SECONDARY_PAGES = 2
_MULTIDOC_MAX_CHUNKS_PER_SECONDARY = 3
_MULTIDOC_MAX_SECONDARY_CHARS = 5000
_MULTIDOC_MIN_SECONDARY_SCORE = 0.5
_MULTIDOC_DUPE_TOKEN_OVERLAP = 0.85


def _multidoc_token_set(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _multidoc_near_duplicate(document, included_token_sets):
    """True when most of `document`'s own tokens already appear inside a
    single already-included chunk (the primary page or a previously
    added secondary chunk) - i.e. the chunk would add little or no NEW
    information and should be dropped. Measures overlap against the
    candidate's OWN token count (not the included chunk's), so a short
    chunk that is a strict subset of a longer included chunk is dropped,
    while a long chunk merely touching a short one is kept."""
    tokens = _multidoc_token_set(document)

    if not tokens:
        return True

    for included in included_token_sets:
        if len(tokens & included) / len(tokens) >= _MULTIDOC_DUPE_TOKEN_OVERLAP:
            return True

    return False


# ------------------------------------------------------------------
# Sprint D - deterministic query rewrite for the FREE-TEXT retrieval
# path only (BM25 + vector). Applied INSIDE hybrid_search() AFTER
# structured_search() has already missed and after the referentless /
# cold-follow-up gates have been evaluated on the RAW question, so:
#   - structured retrieval, the domain gate, contextual-reference
#     resolution, and answer generation all keep the raw question;
#   - only the lexical search the user's words feed sees the expansion.
#
# Every rule is a high-precision trigger (a specific acronym token or a
# specific casual-intent phrase); the expansion APPENDS canonical WLU
# vocabulary the corpus actually uses, keeping the user's own words
# first so BM25/embedding still weigh the original intent highest. A
# query that matches no rule is returned byte-for-byte identical, so
# the 202-item benchmark and regression suite are untouched for every
# question not explicitly targeted (verified: exactly three benchmark
# items fire any rule - FAQ_002 "MSW" (expansion "Master of Social
# Work" is exactly what MSW means), and STUDENTSVC_001/011 (support-
# context international questions whose "support services" expansion
# reinforces the support-and-wellness page they already require). No
# decline-expected item fires any rule.
#
# No LLM, no external API, no invented facts - expansions are static
# ground-truth terms. Guards on every rule skip the append when the
# expansion's core phrase is already present, so an already well-formed
# query is never duplicated.
# ------------------------------------------------------------------

# (acronym token, canonical phrase, extra terms) - token is matched as a
# standalone word-boundary token, case-insensitively, so course codes
# like "CS100" (no boundary) are never touched.
# Every expansion is a tuple of phrases; _rewrite_query's _append()
# helper adds only phrases not already present in the accumulated query,
# so rules never duplicate vocabulary the query (or an earlier rule) has
# already supplied (e.g. "Who does research in AI" must not repeat
# "artificial intelligence" after the acronym rule added it).
_QR_ACRONYMS = [
    ("ai", "artificial intelligence", "machine learning"),
    ("ml", "machine learning", None),
    ("nlp", "natural language processing", None),
    ("cs", "computer science", None),
    ("msw", "master of social work", None),
]

_QR_STRESS_PATTERN = re.compile(
    r"\b(stress|stressed|stressing|anxious|anxiety|overwhelmed|overwhelming|"
    r"depressed|depression|suicidal|suicide)\b"
    r"|feeling\s+(down|low|bad|blue)"
    r"|not\s+(ok|okay|alright|fine)",
    re.IGNORECASE,
)
_QR_STRESS_EXPANSION = (
    "mental health", "wellness", "counselling", "student support services",
)

# Casual graduate-school intent ("I want a Masters", "looking to get a
# graduate degree"), NOT "Master of Finance"-style proper names (those
# route through structured program search and match neither alternative:
# no subject pronoun before the degree word, no "master <degree>"
# adjacency).
_QR_GRADUATE_PATTERN = re.compile(
    r"\b(i|i'?m|we|we'?re|looking\s+to|want(ing)?\s+to|wanna|gonna|"
    r"going\s+to|how\s+(do|can|should)\s+i|thinking\s+about)\b"
    r"[^.!?]{0,45}?\b(masters?|graduate|grad)\b"
    r"|\b(masters?|graduate|grad)\s+(degree|program|studies|school|admissions?)\b",
    re.IGNORECASE,
)
_QR_GRADUATE_EXPANSION = (
    "graduate programs", "master programs", "admissions", "graduate studies",
)

_QR_SUPERVISOR_PATTERN = re.compile(
    r"\bfind\s+(?:a\s+|my\s+|the\s+|a\s+research\s+)?supervisor\b"
    r"|\bresearch\s+supervisor\b"
    r"|\bsupervisor\s+(?:for|in)\b",
    re.IGNORECASE,
)
_QR_SUPERVISOR_EXPANSION = ("faculty research", "graduate studies")

# AI-term + faculty/research-term co-occurrence in EITHER order, without
# crossing a sentence boundary. Both alternatives use the same
# faculty/research alternation (with plural stems), so "research in AI",
# "AI professors" and "professors of machine learning" all fire.
_QR_AI_FACULTY_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|machine learning)\b"
    r"[^.!?]{0,25}?"
    r"\b(professors?|facult(?:y|ies)|researchers?|research(?:es|ing)?|supervisors?)\b"
    r"|\b(professors?|facult(?:y|ies)|researchers?|research(?:es|ing)?|supervisors?)\b"
    r"[^.!?]{0,25}?"
    r"\b(ai|artificial intelligence|machine learning)\b",
    re.IGNORECASE,
)
_QR_AI_FACULTY_EXPANSION = (
    "faculty research", "computer science",
)

# International students in a SUPPORT/HELP context only. A bare
# "tuition / fees / how to apply for international students" query is
# precise as-is and must not pick up "support services" noise - the
# support-and-wellness vocabulary only helps when the user is actually
# asking about support (benchmark STUDENTSVC_001/011, MG_INT1).
_QR_INTERNATIONAL_PATTERN = re.compile(
    r"\binternational\s+students?\b", re.IGNORECASE
)
_QR_INTERNATIONAL_SUPPORT_CONTEXT = re.compile(
    r"\b(support|help|service|services|assistance|resource|resources|"
    r"immigration|visa|counselling|wellness|mental\s+health)\b",
    re.IGNORECASE,
)
_QR_INTERNATIONAL_EXPANSION = ("support services",)


def _rewrite_query(question):
    """Return the expanded retrieval query, or the original question
    byte-for-byte when no rule fires. Pure function - callers keep the
    raw question for every non-retrieval purpose."""
    original = question
    parts = [question]
    q_lower = " ".join(parts).lower()

    def _append(phrases):
        # Add only phrases not already present, keeping the user's own
        # words first so the original intent always dominates BM25/vector
        # weighting.
        nonlocal q_lower
        for phrase in phrases:
            if phrase.lower() not in q_lower:
                parts.append(phrase)
                q_lower = " ".join(parts).lower()

    # 1. Acronym expansion (standalone word-boundary token).
    for token, canonical, extra in _QR_ACRONYMS:
        if re.search(rf"\b{re.escape(token)}\b", q_lower):
            _append([p for p in (canonical, extra) if p])

    # 2. Stress / mental-health intent.
    if _QR_STRESS_PATTERN.search(q_lower):
        _append(_QR_STRESS_EXPANSION)

    # 3. Graduate intent (skip when the query already names a graduate
    #    program, i.e. is already well-formed).
    if _QR_GRADUATE_PATTERN.search(q_lower) and not re.search(
        r"\bgraduate\s+program", q_lower
    ):
        _append(_QR_GRADUATE_EXPANSION)

    # 4. Research supervisor.
    if _QR_SUPERVISOR_PATTERN.search(q_lower):
        _append(_QR_SUPERVISOR_EXPANSION)

    # 5. AI faculty / research (skip when already explicit).
    if _QR_AI_FACULTY_PATTERN.search(q_lower) and not (
        "computer science" in q_lower and "faculty" in q_lower
    ):
        _append(_QR_AI_FACULTY_EXPANSION)

    # 6. International students in a support/help context only.
    if (
        _QR_INTERNATIONAL_PATTERN.search(q_lower)
        and _QR_INTERNATIONAL_SUPPORT_CONTEXT.search(q_lower)
    ):
        _append(_QR_INTERNATIONAL_EXPANSION)

    rewritten = " ".join(parts)

    return rewritten if rewritten != original else original


# ------------------------------------------------------------------
# Sprint E - Intent Planner & Knowledge Aggregation (free-text path
# only). Applied INSIDE hybrid_search() AFTER structured_search() has
# already missed, after the referentless / cold-follow-up gates, after
# the Sprint D rewrite, and after the winner page + Sprint C secondary
# sources have been chosen - so:
#   - structured retrieval, the domain gate, contextual-reference
#     resolution, winner selection, the distance gate, and the citation
#     pipeline are ALL untouched (the primary source stays byte-
#     identical);
#   - the layer only ADDS labeled, grounded WLU facet pages to the
#     multi-source context the answer generator already receives.
#
# The Planner is deterministic (regex triggers, no LLM). Each intent
# maps to a small set of FACETS - canonical WLU pages that answer the
# question from a complementary angle (e.g. "I want a Masters" ->
# graduate admissions + funding + studies overview; "I'm stressed" ->
# mental-health resources + urgent care + wellness services). Facet
# pages are static ground truth: each is a known URL fragment in the
# corpus, fetched through the existing BM25 index
# (bm25_search_in_section), so no new embeddings, no external API, no
# invented facts. The relevance gate is the intent trigger itself (the
# cross-encoder is uncalibrated for long boilerplate-heavy chunks vs.
# casual questions - probing showed clearly-relevant facet pages
# scoring negative), backed by three hard safety filters:
#   - a facet page is skipped if its URL is already in the source set
#     (winner or a Sprint C secondary) - never duplicated;
#   - every facet chunk must survive the Sprint C near-duplicate check
#     against everything already included - never repeated content;
#   - every facet must actually exist in the corpus (graceful no-op).
#
# A query that fires no intent, or whose facet pages all dedupe/skip,
# leaves the context byte-identical to the pre-Sprint-E output.
# ------------------------------------------------------------------

# Per-intent conditional facet triggers. `None` = the facet fires for
# every question that fires its intent; otherwise the regex must also
# match (e.g. the urgent-care page only joins a wellness answer when
# the question actually signals crisis, and the working-in-Canada page
# only joins an international answer when work is mentioned).
_IP_INTERNATIONAL_MENTION = re.compile(r"\binternational\b", re.IGNORECASE)
_IP_WELLNESS_CRISIS = re.compile(
    r"\b(crisis|urgent|emergency|after[- ]?hours|suicid|988|24/7|24 ?7|immediate)\w*",
    re.IGNORECASE,
)
_IP_INTL_IMMIG = re.compile(r"\b(immigrat|visa|permit|status)\w*", re.IGNORECASE)
_IP_INTL_WORK = re.compile(
    r"\b(work(?!shop)|job|employment|co.?op|internship|career)\w*", re.IGNORECASE,
)
_IP_INTL_MOVE = re.compile(
    r"\b(move|arriv|travel|flight|airport|accommod|housing|settle|land|new to canada|come to canada)\w*",
    re.IGNORECASE,
)
_IP_INTL_FINANCE = re.compile(
    r"\b(tuition|financ|pay|cost|money|expense|fee|budget)\w*", re.IGNORECASE,
)
_IP_WRITING_APPT = re.compile(
    r"\b(appointment|book|meet|schedule|consult|drop[ -]?in|one[ -]?on[ -]?one|sign.?up)\w*",
    re.IGNORECASE,
)
_IP_WRITING_RES = re.compile(
    r"\b(resource|handout|guide|citation|apa|mla|chicago|tip|example|template|sample|format|worksheet)\w*",
    re.IGNORECASE,
)

# (id, pattern, negative, facets[(label, url_fragment, when_or_None)]).
# Facets are ordered most-important-first; aggregation stops after
# _IP_MAX_FACET_PAGES page-groups are added.
_IP_GRADUATE_PATTERN = re.compile(
    r"\bmasters?('|s)?\b|\bmaster of\b|\bpostgraduate\b|\bpost-graduate\b|"
    r"\bgraduate\b|\bgrad\b|\bph\.?d\b|\bdoctorate\b|\bdoctoral\b",
    re.IGNORECASE,
)
_IP_GRADUATE_NEGATIVE = re.compile(
    r"\bcourses?\b|\bclasses?\b|\bprerequisit\w*\b", re.IGNORECASE,
)
_IP_WELLNESS_PATTERN = re.compile(
    r"\bstress(?:ed|ful|ing)?\b|\banxious?\b|\banxiety\b|\bdepress(?:ed|ion)?\b|"
    r"\bsuicid\w*\b|\bmental health\b|\bcounsell\w*\b|\bcounsel(?!ing)\w*\b|"
    r"\bwell[- ]?being\b|\bwellbeing\b|\bwellness\b|\btherap\w*\b|\bpsycholog\w*\b|"
    r"\bpsychiatr\w*\b|\bcrisis\b|\bstruggl\w*\b|\boverwhelm\w*\b|\bemotional\b|\bself[- ]?care\b",
    re.IGNORECASE,
)
_IP_INTERNATIONAL_PATTERN = re.compile(
    r"\binternational students?\b|\bimmigration\b|\bstudy permit\b|\bvisa\b|"
    r"\bnew to canada\b|\bcome to canada\b|\bmove to (?:canada|l(?:a)?urier)\b|"
    r"\barriv\w* (?:in|to|at) (?:canada|l(?:a)?urier)\b",
    re.IGNORECASE,
)
_IP_INTERNATIONAL_CONTEXT = re.compile(
    r"\bsupport\b|\bhelp\b|\bimmigration\b|\bvisa\b|\bpermit\b|\bcounsell\w*\b|"
    r"\binsurance\b|\bmove\b|\barriv\w*\b|\bwork(?!shop)\w*\b|\bjob\b|\bstudy\b|"
    r"\bhousing\b|\baccommod\w*\b|\badvise?\w*\b|\bconnect\w*\b|\borient\w*\b|"
    r"\btuition\b|\bfinanc\w*\b|\bpay\b|\bexpense\w*\b|\bnew to canada\b|"
    r"\bcome to canada\b|\bregister\w*\b|\benroll\w*\b",
    re.IGNORECASE,
)
_IP_WRITING_PATTERN = re.compile(
    r"\bwriting (?:centre|center|support|help|services?|appointments?|programs?|"
    r"workshops?|resources?|assist\w*)\b|"
    r"\bessay (?:help|writing|support|assist\w*)\b|"
    r"\bhelp (?:with|me with)? ?(?:my )?(?:essay|paper|assignment|writing|thesis)\b|"
    r"\b(?:improve|better|polish) (?:my )?(?:writing|essay|paper)\b|\bproofread\w*\b|"
    r"\b(?:citation|apa|mla|chicago|academic writing)\b.{0,30}?\b(?:help|guide|format|style)\b|"
    r"\bthesis (?:writing|support)\b",
    re.IGNORECASE,
)

# Direct Writing Centre ENTITY queries ("Where is the Writing Centre?",
# "Writing Services hours", "writing support location") name the service
# itself rather than one of its activities, so the appointment/location/
# booking/email detail lives behind every conditional writing facet. When
# this fires, the conditional facets' `when` gates are bypassed so the
# full set (programs + appointments + resources) is aggregated even for a
# query with no appointment/resource token - the physical location of the
# Writing Centre (One Market OM207 / Peters Building, 2nd floor P226) is
# on the appointments page and was being missed exactly this way.
_IP_WRITING_CENTRE_QUERY = re.compile(
    r"\bwriting (?:centre|center|services?|support)\b|"
    r"\bwhere (?:is|are)\b.{0,25}?\bwriting\b",
    re.IGNORECASE,
)

# Convocation intent (Sprint 2A, BUG6): "When is Fall 2026 convocation?"
# was answered from the general Important Dates page, which surfaces only
# the Waterloo dates - the ceremonies page's own Fall 2026 table (with the
# separate Brantford ceremony) lost the winner slot. Aggregating the
# ceremonies page as a facet puts the complete date table in context and
# makes "For Brantford?" resolvable (the campus qualifier reuses the
# topic entity recorded below when this fires).
_IP_CONVOCATION_PATTERN = re.compile(
    r"\bconvocation\b",
    re.IGNORECASE,
)

_IP_INTENTS = [
    {
        "id": "graduate",
        "pattern": _IP_GRADUATE_PATTERN,
        "negative": _IP_GRADUATE_NEGATIVE,
        "facets": [
            # (label, canonical URL fragment, when-or-None)
            ("Graduate Admissions & Requirements",
             "graduate-and-postdoctoral-studies/admissions", None),
            ("Graduate Funding & Awards",
             "graduate-funding-and-awards/index", None),
            ("Graduate Studies",
             "graduate-and-postdoctoral-studies/index", None),
            ("International Graduate Students",
             "graduate-and-postdoctoral-studies/international-students",
             _IP_INTERNATIONAL_MENTION),
        ],
    },
    {
        "id": "wellness",
        "pattern": _IP_WELLNESS_PATTERN,
        "facets": [
            ("Mental Health Resources", "mental-health-resources", None),
            ("Student Wellness Centre Services",
             "student-wellness-centre/services", None),
            ("Urgent & After-Hours Care", "urgent-care", _IP_WELLNESS_CRISIS),
        ],
    },
    {
        "id": "international",
        "pattern": _IP_INTERNATIONAL_PATTERN,
        "context": _IP_INTERNATIONAL_CONTEXT,
        "facets": [
            ("Immigration & Visas",
             "international-student-support/immigration/index", _IP_INTL_IMMIG),
            ("Working in Canada",
             "international-student-support/working-in-canada", _IP_INTL_WORK),
            ("Planning Your Move",
             "international-student-support/planning-your-move", _IP_INTL_MOVE),
            ("International Finances",
             "international-student-support/assets/resources/"
             "paying-tuition-and-managing-your-expenses", _IP_INTL_FINANCE),
            ("International Student Support",
             "international-student-support/index", None),
        ],
    },
    {
        "id": "writing",
        "pattern": _IP_WRITING_PATTERN,
        "when_override": _IP_WRITING_CENTRE_QUERY,
        "facets": [
            ("Writing Support Programs",
             "writing/writing-support-programs", None),
            ("Writing Appointments",
             "student-success/appointments", _IP_WRITING_APPT),
            ("Writing Resources & Handouts",
             "student-success/resources", _IP_WRITING_RES),
        ],
    },
    {
        "id": "convocation",
        "pattern": _IP_CONVOCATION_PATTERN,
        # The ceremony date table is the entire point of this facet, so
        # it must not be silently lost when the ceremonies page is
        # already present as a Sprint C secondary - the secondary
        # mechanism caps how many of its chunks survive (chunk count +
        # total secondary-char budget), and with a busy pool that cap
        # drops the Brantford row while keeping the Waterloo rows (the
        # Brantford chunk ranks below the Waterloo/exceptions chunks on
        # a generic "convocation" query). force_page makes the facet
        # aggregation run for this page even when its URL is already in
        # included_urls; the near-duplicate filter then naturally drops
        # the chunks already in context and adds only the missing ones
        # (the Brantford ceremony row), so no content is duplicated.
        "force_page": True,
        "facets": [
            ("Convocation Ceremonies",
             "convocation/ceremonies/index.html", None),
        ],
    },
]

_IP_MAX_FACET_PAGES = 3
_IP_MAX_CHUNKS_PER_FACET = 3
_IP_MAX_FACET_CHARS = 9000
_IP_FACET_TOP_K = 4


def _plan_intent(question):
    """Deterministic intent planner. Returns the first intent whose
    trigger fires (and whose negative guard does not), or None."""
    q = question.lower()
    for intent in _IP_INTENTS:
        if not intent["pattern"].search(q):
            continue
        if intent.get("negative") and intent["negative"].search(q):
            continue
        if intent.get("context") and not intent["context"].search(q):
            continue
        return intent
    return None


def intent_id(question):
    """Public wrapper for the intent planner: the id of the Sprint E
    intent that fires for `question` ('graduate', 'wellness',
    'international', 'writing', 'convocation'), or None. Used by app.py's
    wellness rescue so the deterministic intent planner can run BEFORE the
    off-topic gate decides how to answer ("I'm stressed." fires the
    wellness intent but fails the keyword/LLM domain check, and would
    otherwise get a social decline instead of WLU wellness resources)."""
    intent = _plan_intent(question)
    return intent["id"] if intent else None


def _aggregate_intent_facets(intent, question, winner_url, included_urls,
                             included_token_sets):
    """Fetch the intent's canonical facet pages as labeled secondary
    sources. Pure retrieval + dedupe - no LLM. Returns a list of groups
    {"label", "url", "title", "chunks"}; empty when no facet qualifies."""
    q_lower = question.lower()
    groups = []
    facet_chars = 0
    included_urls = set(included_urls)

    # A direct entity query ("Where is the Writing Centre?") names the
    # service, not one of its activities - every conditional facet's
    # `when` gate is bypassed so the full detail set (which for the
    # Writing Centre includes the physical location on the appointments
    # page) reaches the context. No-op for every other intent/query.
    when_override = intent.get("when_override")
    override_fires = (
        when_override is not None and when_override.search(q_lower)
    )

    for label, url_fragment, when in intent["facets"]:

        if when is not None and not when.search(q_lower) and not override_fires:
            continue

        if len(groups) >= _IP_MAX_FACET_PAGES:
            break

        candidates = hybrid_rerank.bm25_search_in_section(
            collection, question, url_fragment, top_k=_IP_FACET_TOP_K,
        )

        page_url = next(
            (c["url"] for c in candidates
             if url_fragment in (c.get("url") or "")),
            None,
        )

        # The canonical page is absent from the corpus (refresh/rename)
        # or already cited (winner / Sprint C secondary) - skip. A facet
        # that sets force_page (convocation) is the exception: the whole
        # point of that facet is the complete date table, and the
        # secondary mechanism may have only kept a subset of the page's
        # chunks (dropping e.g. the Brantford ceremony row). Forcing the
        # aggregation to run anyway is safe because the near-duplicate
        # filter below drops every chunk already in context and only the
        # missing ones survive.
        if page_url is None:
            continue
        if page_url in included_urls and not intent.get("force_page"):
            continue

        page_chunks = [
            c["document"] for c in candidates
            if (c.get("url") or "") == page_url
        ][:_IP_MAX_CHUNKS_PER_FACET]

        if not page_chunks:
            continue

        kept = []

        for document in page_chunks:

            document = _strip_known_boilerplate_text(document)

            if _multidoc_near_duplicate(document, included_token_sets):
                continue

            if facet_chars + len(document) > _IP_MAX_FACET_CHARS:
                continue

            kept.append(document)
            facet_chars += len(document)
            included_token_sets.append(_multidoc_token_set(document))

        if not kept:
            continue

        included_urls.add(page_url)

        groups.append({
            "label": label,
            "url": page_url,
            "title": candidates[0].get("title"),
            "chunks": kept,
        })

    return groups


def hybrid_search(question, memory=None):

    result = structured_search(question, memory)

    if result:

        hybrid_rerank.record_debug_trace({
            "question": question,
            "structured_retrieval_used": True,
        })

        return result

    if _is_referentless_query(question):

        hybrid_rerank.record_debug_trace({
            "question": question,
            "structured_retrieval_used": False,
            "gate_passed": False,
            "referentless_query": True,
        })

        return (_GENERIC_CLARIFICATION_MESSAGE, None, "not_found")

    # Cold-start follow-up phrase with no established memory context
    # (see _FOLLOWUP_NO_CONTEXT_MESSAGE's comment): same rationale as
    # the referentless gate above, but for the FOLLOWUP_PHRASES set
    # rather than reference markers. Guarded on the shared
    # _memory_has_any_context() so a follow-up AFTER a real turn keeps
    # working exactly as before - structured_search()'s own FOLLOWUP
    # MEMORY rewrite has already failed above precisely because there
    # is no entity to substitute, so reaching this point means the
    # phrase genuinely has nothing to refer to.
    if (
        (memory is None or not _memory_has_any_context(memory))
        and normalize_followup_text(question) in FOLLOWUP_PHRASES
    ):

        hybrid_rerank.record_debug_trace({
            "question": question,
            "structured_retrieval_used": False,
            "gate_passed": False,
            "cold_followup_no_context": True,
        })

        return (_FOLLOWUP_NO_CONTEXT_MESSAGE, None, "not_found")

    # VECTOR SEARCH

    # Sprint D - deterministic query rewrite for this retrieval path only
    # (see _rewrite_query's design comment). `retrieval_question` is a
    # pure function of the RAW question; when no rule fires it is
    # byte-identical and the whole Sprint-D layer is a no-op. Every
    # consumer outside this function (structured search, domain gate,
    # canonical-section detection, answer prompt, citations) keeps the
    # raw `question` - only the lexical search the user's words feed is
    # allowed to see the expansion.
    retrieval_question = _rewrite_query(question)
    query_rewritten = retrieval_question != question

    # Stage 1 - the RAW question is always embedded first and its own
    # nearest neighbour gates. The distance threshold was calibrated
    # against raw-query distances (see the calibration note below), so an
    # unrewritten query gates EXACTLY as it did pre-Sprint-D; the rewrite
    # can later rebuild the candidate POOL, never the gate's yes/no basis.
    raw_results = search_vector(question)

    # Restores the ORIGINAL calibration assumption the 1.19/1.23 data
    # points (see _VECTOR_SEARCH_DISTANCE_THRESHOLD's own comment) were
    # actually measured against: the single best raw dense match, from
    # before any reranking existed at all (back when search_vector()
    # fetched only 5 results and the gate checked their raw top-1).
    # Chroma's .query() already orders results by ascending distance, so
    # the minimum distance in this wider 15-candidate pool is exactly
    # the closest neighbour in the whole collection - identical to what
    # that narrower, unreranked top-1 query would have returned,
    # regardless of how many candidates are actually requested.
    #
    # This was NOT what the gate checked before this fix.
    # _rerank_vector_candidates() (metadata-aware reranking: title/URL
    # keyword bonus + boilerplate/news penalty, added later to improve
    # WHICH page gets shown/cited) had been repurposed as the gate's
    # candidate source too, so a page whose title/URL merely shared a
    # word with the question could outscore every genuinely close match
    # and become the one distance checked against the threshold - even
    # with a truly relevant, in-threshold page sitting right next to it
    # in the same pool. Confirmed live: "Where is the Writing Centre?"
    # was declined this way - the reranker's winner (title: "Accessible
    # Learning Centre", matching "Centre" and "Writing" by coincidence)
    # sat at distance 1.221 (over threshold), while a genuinely relevant
    # page was sitting in the same pool at distance 1.098 (comfortably
    # under it) and never got considered by the gate at all.
    #
    # BM25, Reciprocal Rank Fusion, and cross-encoder reranking below are
    # completely unaffected by this - they still decide WHICH page
    # answers the question once the gate has approved it; this only
    # restores the gate's own yes/no input to its original basis.
    raw_top_distance = min(raw_results["distances"][0])

    # Exempt recognized follow-up phrases ("tell me more", "explain",
    # ...): these carry no semantic content of their own to embed - the
    # literal phrase itself always retrieves weakly - so the gate would
    # otherwise misread a normal conversational continuation (already
    # relying on chat history, not fresh retrieval, for its answer - the
    # same reason app.py's off-topic gate exempts these) as a low-
    # confidence "not found" case.
    is_followup = normalize_followup_text(question) in FOLLOWUP_PHRASES

    if raw_top_distance > _VECTOR_SEARCH_DISTANCE_THRESHOLD and not is_followup:

        # Stage 2 - rescue attempt. The RAW query declined the gate but a
        # rewrite rule fired: try the expanded query's own nearest
        # neighbour and accept only if THAT is in-threshold. This is the
        # "expand only when beneficial" guarantee - a rewrite can never
        # turn an already-answerable query into not_found (the raw query
        # was going to be declined anyway), and it can only rescue a
        # query the raw form would have declined.
        if query_rewritten:
            rescue_results = search_vector(retrieval_question)
            rescue_top_distance = min(rescue_results["distances"][0])

            if rescue_top_distance <= _VECTOR_SEARCH_DISTANCE_THRESHOLD:
                results = rescue_results
                top_distance = rescue_top_distance
                gate_rescued = True
            else:
                results = raw_results
                top_distance = raw_top_distance
                gate_rescued = False
        else:
            results = raw_results
            top_distance = raw_top_distance
            gate_rescued = False

        if not gate_rescued:

            hybrid_rerank.record_debug_trace({
                "question": question,
                "structured_retrieval_used": False,
                "gate_passed": False,
                "gate_top_distance": top_distance,
                "query_rewritten": query_rewritten,
                "rewritten_query": retrieval_question if query_rewritten else question,
            })

            return _NO_CONFIDENT_MATCH_MESSAGE, None, "not_found"

        # Accepted via rescue - fall through with the rewritten pool.

    else:

        # Raw query passed (or is an exempt follow-up continuation). The
        # gate keeps the RAW query as its basis; only when a rewrite
        # actually fired is the candidate pool rebuilt from the expanded
        # query, so the better terms drive BM25/vector selection too.
        top_distance = raw_top_distance
        gate_rescued = False

        if query_rewritten:
            results = search_vector(retrieval_question)
        else:
            results = raw_results

    # Gate passed - true hybrid retrieval now picks the best page: dense
    # + BM25 candidates (chunk-level, not the gate's per-URL-deduped
    # view), merged by Reciprocal Rank Fusion, reranked by a cross-
    # encoder. hybrid_rerank.bm25_search transparently rebuilds its index
    # whenever the underlying ChromaDB collection's chunk count changes
    # (e.g. after refresh_pipeline.py), so no app restart is needed for
    # it to pick up refreshed content.
    dense_candidates = hybrid_rerank.dense_candidates_from_results(results)

    bm25_candidates = hybrid_rerank.bm25_search(collection, retrieval_question)

    fused = hybrid_rerank.reciprocal_rank_fusion(dense_candidates, bm25_candidates)

    # Canonical-section intent (see _SECTION_INTENT_PATTERNS /
    # _apply_canonical_section_preference): when the question names a
    # service/administrative section and none of that section's pages
    # made it into the fused pool, merge the section's top BM25-scoring
    # chunks directly (a section-restricted search over the same BM25
    # index - no fresh Chroma query) so the section boost below has a
    # target. Without this, e.g. the disability-justice-and-accessibility
    # page behind STUDENTSVC_004/014 is never retrieved by the dense/BM25
    # top-k at all. Merged chunks are pinned just above the pool's current
    # top fused score so the cross-encoder still judges them on merit.
    section = _match_canonical_section(question)

    if section is not None and not any(
        section in (candidate.get("url") or "") for candidate in fused
    ):

        section_candidates = hybrid_rerank.bm25_search_in_section(
            collection, question, section,
            top_k=_CANONICAL_SECTION_MERGE_TOP_K,
        )

        base_score = fused[0]["fused_score"] if fused else 0.0

        for candidate in section_candidates:
            candidate["fused_score"] = base_score + 0.001
            fused.append(candidate)

    if section is None:

        # No section intent - the EXACT pre-Sprint4 pipeline (top-k
        # cross-encoder rerank, then penalties/boosts), so no question
        # that names none of the canonical sections changes behavior at
        # all (regression guard).
        reranked = hybrid_rerank.cross_encoder_rerank(question, fused)

    else:

        # Section intent - score the FULL fused pool instead of the top-k
        # truncation, because the cross-encoder truncation at RERANK_TOP_K
        # would otherwise hide ranks 11+ from every subsequent boost (e.g.
        # DEADLINE_016's important-dates page sat in the pool at rank 11+
        # and lost on raw score alone). The established penalties/boosts
        # and the section preference below then run over the whole pool.
        reranked = hybrid_rerank.cross_encoder_rerank(question, fused, top_k=None)

    # See _apply_topical_mismatch_penalty()'s own comment: the cross-
    # encoder alone confidently picked a narrow, topically-mismatched
    # page for two confirmed real questions (a program-specific FAQ for
    # a general procedural question; a non-credit-module self-
    # registration guide for an ambiguous "MAC courses" question) -
    # this demotes those specific, confirmed patterns without touching
    # the cross-encoder's judgment for every other query.
    reranked = _apply_topical_mismatch_penalty(
        reranked, _significant_question_words(question)
    )

    # See _apply_faq_intent_boost(): when the question explicitly asks for
    # "the FAQ(s)" for a topic, a candidate whose URL/title marks it as an
    # FAQ page wins by construction - the cross-encoder (and the topical-
    # mismatch penalty above) do not reliably identify the FAQ page for a
    # question that names its topic. No-op for every non-FAQ question.
    reranked = _apply_faq_intent_boost(reranked, question)

    # See _apply_canonical_section_preference(): when the question names
    # a canonical service/administrative section, a candidate from that
    # section wins by preference - the cross-encoder does not know WLU's
    # section taxonomy and demonstrably picks wrong-section winners for
    # deadlines/campus/student-services questions (see its comment). No-op
    # for every question that names no canonical section or has FAQ intent.
    reranked = _apply_canonical_section_preference(reranked, question)

    if section is not None:
        # Restore the cross-encoder's own top-k contract. The same-page
        # context loop below iterates the full `fused` pool (including any
        # merged section candidates), so the winner's chunks are still
        # found after this truncation.
        reranked = reranked[:hybrid_rerank.RERANK_TOP_K]

    winner = reranked[0]

    winner_url = winner["url"]

    # Sprint C - multi-document context construction. The winning page's
    # own chunk(s) in the fused pool (ranked by fused score, the exact
    # pre-Sprint-C content) remain the PRIMARY source of the context and
    # of the citation. Then a bounded number of COMPLEMENTARY chunks
    # from the best OTHER pages in the reranked pool are appended, each
    # surviving a near-duplicate check against everything already
    # included - so the context reads as one synthesized multi-source
    # answer rather than a single page's slice or repeated chunks.
    seen_chunk_ids = set()
    same_page_documents = []

    for candidate in fused:

        if candidate["url"] != winner_url or candidate["id"] in seen_chunk_ids:
            continue

        seen_chunk_ids.add(candidate["id"])
        same_page_documents.append(candidate["document"])

    primary_documents = [
        _strip_known_boilerplate_text(document) for document in same_page_documents
    ]

    # Cross-page near-duplicate suppression pool, seeded with the primary
    # page's content so a secondary chunk that merely restates a primary
    # chunk is dropped (e.g. the same "golden rules" published on several
    # academic-integrity resource pages).
    included_token_sets = [
        _multidoc_token_set(document) for document in primary_documents
    ]

    secondary_groups = []
    secondary_chars = 0

    for candidate in reranked[1:]:

        if candidate["url"] == winner_url:
            continue

        # Absolute relevance floor: the cross-encoder separates genuinely
        # relevant pages (positive scores, typically > 0.5) from weakly
        # or negatively scored noise that the winner's own page outranks.
        # Confirmed live: for a query whose whole pool scored negatively
        # (e.g. a title-word coincidence winner), NO secondary page is
        # added and the context falls back to the single-source form.
        if candidate.get("cross_encoder_score", -1.0) < _MULTIDOC_MIN_SECONDARY_SCORE:
            continue

        group = next(
            (g for g in secondary_groups if g["url"] == candidate["url"]),
            None,
        )

        if (
            group is not None
            and len(group["chunks"]) >= _MULTIDOC_MAX_CHUNKS_PER_SECONDARY
        ):
            continue

        document = _strip_known_boilerplate_text(candidate["document"])

        if _multidoc_near_duplicate(document, included_token_sets):
            continue

        if secondary_chars + len(document) > _MULTIDOC_MAX_SECONDARY_CHARS:
            continue

        if group is None:

            if len(secondary_groups) >= _MULTIDOC_MAX_SECONDARY_PAGES:
                continue

            group = {
                "url": candidate["url"],
                "title": candidate.get("title"),
                "chunks": [],
            }
            secondary_groups.append(group)

        group["chunks"].append(document)
        secondary_chars += len(document)
        included_token_sets.append(_multidoc_token_set(document))

    # Labeled multi-source context: the primary page first (content
    # unchanged, wrapped in a Source header), then each complementary
    # page (Sprint C), then each intent facet page (Sprint E). Labels
    # carry the page TITLES / facet labels only - URLs are deliberately
    # not injected into the prompt (the citation rendered below the
    # answer already shows them), so the LLM never echoes raw links.
    sections = [f"Source 1: {winner.get('title') or 'Primary page'}"]
    sections.extend(primary_documents)

    for index, group in enumerate(secondary_groups, start=2):
        sections.append(f"Source {index}: {group['title'] or 'Related page'}")
        sections.extend(group["chunks"])

    source_urls = [winner_url] + [group["url"] for group in secondary_groups]

    # Sprint E - Intent Planner & Knowledge Aggregation. Appends
    # intent-specific canonical facet pages to whichever context form is
    # being built. Every facet is grounded WLU content pulled through
    # the existing BM25 index, labeled by its facet so the answer
    # generator can organize a multi-angle answer. When no intent fires
    # or every facet dedupes, `facet_groups` stays empty and the
    # context/source above are returned byte-identical to Sprint C.
    facet_groups = []
    intent = _plan_intent(question)

    if intent is not None:
        facet_groups = _aggregate_intent_facets(
            intent, question, winner_url,
            source_urls, included_token_sets,
        )

    # Sprint 2A (BUG6) - record a "topic" entity for vector-answered
    # intent queries so a follow-up can resolve against it. The campus
    # qualifier in resolve_contextual_reference ("For Brantford?", "At
    # Waterloo?") rewrites "<topic> <campus>" and answers through
    # hybrid_search - without this write the follow-up had no established
    # context to reuse (entity_history was empty after a convocation
    # turn) and was answered fresh, ignoring the prior question.
    if intent is not None and intent["id"] == "convocation":
        _record_entity(
            memory, "topic", "convocation", "Convocation", "hybrid_search",
        )

    for index, group in enumerate(
        facet_groups, start=1 + len(secondary_groups) + 1,
    ):
        sections.append(f"Source {index}: {group['label']}")
        sections.extend(group["chunks"])
        source_urls.append(group["url"])

    if not secondary_groups and not facet_groups:

        # No complementary source at all - EXACTLY the pre-Sprint-C
        # context (byte-identical) and a single-string source, so every
        # question with no useful secondary page behaves as before.
        context = "\n\n".join(primary_documents)
        source = winner_url

    else:

        context = "\n\n".join(sections)
        source = source_urls

    hybrid_rerank.record_debug_trace({
        "question": question,
        "structured_retrieval_used": False,
        "gate_passed": True,
        "gate_top_distance": top_distance,
        "query_rewritten": query_rewritten,
        "rewritten_query": retrieval_question if query_rewritten else question,
        "dense_candidates": dense_candidates,
        "bm25_candidates": bm25_candidates,
        "fused_ranking": fused,
        "cross_encoder_scores": reranked,
        "final_selected_chunk": winner,
        "secondary_sources": [
            {"url": g["url"], "title": g["title"], "n_chunks": len(g["chunks"])}
            for g in secondary_groups
        ],
        "intent_planner": intent["id"] if intent is not None else None,
        "intent_facet_sources": [
            {"label": g["label"], "url": g["url"], "n_chunks": len(g["chunks"])}
            for g in facet_groups
        ],
    })

    return context, source, "vector"