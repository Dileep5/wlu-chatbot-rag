import re

import streamlit as st

# Phase 13B: coarse response_type -> renderer grouping. This is a safe
# refactor only - these groups decide which of the four renderers below
# handles a given response_type, but every renderer still shows exactly
# the same answer/source text as before (Phase 13A). Anything not listed
# here (department_profile, research, vector, and the None used by
# greeting/conversation/off-topic/clarification paths) falls through to
# render_generic(), which is byte-identical to today's plain rendering.
_COURSE_RESPONSE_TYPES = {
    "course",
    "prerequisite",
    "course_instructors",
}

_FACULTY_RESPONSE_TYPES = {
    "faculty_profile",
    "faculty_list",
    "faculty_topic_courses",
    "department_faculty_list",
}

_PROGRAM_RESPONSE_TYPES = {
    "program",
    "undergraduate_requirements",
    "graduate_requirements",
    "undergraduate_program_list",
    "coordinator",
}


def _render_source(source):

    if source:
        st.markdown(f"**Source:** {source}")


# Phase 13C: known labels the underlying context text can contain
# (search_course()'s own f-string in retriever.py, plus the Sprint 10D
# metadata section) - used as lookahead boundaries so a field's value
# is captured up to whichever label comes next, not a fixed line count.
# This list is only used to recognize where one field's value ends, not
# to change what retriever.py produces.
_COURSE_FIELD_LABELS = [
    "Course Code",
    "Course Name",
    "Credits",
    "Department",
    "Level",
    "Description",
    "Prerequisites",
    "Corequisites",
    "Exclusions",
    "Location",
    "Notes",
]

_COURSE_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _COURSE_FIELD_LABELS
)

# Fields shown on the card, in display order (Requirement 3) - not the
# same order they appear in the source text.
_COURSE_CARD_FIELDS = [
    "Course Code",
    "Course Name",
    "Credits",
    "Prerequisites",
    "Description",
]


def _extract_course_field(label, text):

    pattern = (
        rf"{re.escape(label)}:\s*(.*?)"
        rf"(?=\n\s*(?:{_COURSE_FIELD_ALTERNATION}):|\Z)"
    )

    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    value = match.group(1).strip()

    return value or None


def _parse_course_fields(answer):
    """Best-effort parse of the existing label:value course text. Returns
    None (parsing "failed") unless both Course Code and Course Name are
    found - the two fields that actually identify a course - since the
    answer text for a "course" response_type is normally LLM-paraphrased
    prose (not the raw labeled context: "course" isn't one of app.py's
    deterministic response_types), so these labels frequently won't be
    present at all. That's expected, not an error - render_course() falls
    back to the original rendering whenever this returns None."""

    code = _extract_course_field("Course Code", answer)
    name = _extract_course_field("Course Name", answer)

    if not code or not name:
        return None

    return {
        "Course Code": code,
        "Course Name": name,
        "Credits": _extract_course_field("Credits", answer),
        "Prerequisites": _extract_course_field("Prerequisites", answer),
        "Description": _extract_course_field("Description", answer),
    }


def _render_course_fallback(answer, source):

    st.markdown(f"📘 Course\n\n{answer}")
    _render_source(source)


# A single st.markdown() call, not one per field - the evaluate.py
# AppTest harness reads only the first markdown element of the last
# chat message, so every bit of card text has to live in that one call
# for the existing regression suite to keep seeing the full answer.
def render_course(answer, source):

    fields = _parse_course_fields(answer)

    if not fields:
        _render_course_fallback(answer, source)
        return

    lines = ["📘 Course", ""]

    for label in _COURSE_CARD_FIELDS:

        value = fields.get(label)

        if value:
            lines.append(f"**{label}**")
            lines.append(value)
            lines.append("")

    st.markdown("\n".join(lines).rstrip())
    _render_source(source)


def render_faculty(answer, source):

    st.markdown(f"👨‍🏫 Faculty\n\n{answer}")
    _render_source(source)


def render_program(answer, source):

    st.markdown(f"🎓 Program\n\n{answer}")
    _render_source(source)


def render_generic(answer, source):

    st.markdown(answer)
    _render_source(source)


def render_response(response_type, answer, source):

    if response_type in _COURSE_RESPONSE_TYPES:
        render_course(answer, source)

    elif response_type in _FACULTY_RESPONSE_TYPES:
        render_faculty(answer, source)

    elif response_type in _PROGRAM_RESPONSE_TYPES:
        render_program(answer, source)

    else:
        render_generic(answer, source)
