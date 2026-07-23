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
# same order they appear in the source text. Corequisites/Exclusions/
# Location/Notes (Phase 13D) come after the original five fields - since
# "course" now bypasses the LLM entirely, these Sprint 10D metadata
# fields would otherwise never reach the user at all (they used to be
# folded into the LLM's paraphrase; the card is now the only rendering
# path, so it has to cover everything the raw text can carry, not just
# the original five).
_COURSE_CARD_FIELDS = [
    "Course Code",
    "Course Name",
    "Credits",
    "Prerequisites",
    "Description",
    "Corequisites",
    "Exclusions",
    "Location",
    "Notes",
]


def _extract_labeled_field(label, text, label_alternation):
    """Captures a label's value up to whichever known label comes next
    (or end of string) - shared by course and faculty parsing, since both
    answer shapes are the same pattern: a fixed set of "Label: value"
    lines, some of which (Description/Biography) continue over multiple
    lines rather than ending at the first newline."""

    pattern = (
        rf"{re.escape(label)}:\s*(.*?)"
        rf"(?=\n\s*(?:{label_alternation}):|\Z)"
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

    fields = {
        label: _extract_labeled_field(label, answer, _COURSE_FIELD_ALTERNATION)
        for label in _COURSE_CARD_FIELDS
    }

    if not fields["Course Code"] or not fields["Course Name"]:
        return None

    return fields


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


# Phase 13E: known labels the raw faculty_profile context can contain
# (search_faculty()'s own f-string in retriever.py) - "Contact" is a bare
# section heading with no value of its own, but still has to be listed
# here so it's recognized as a boundary when capturing the field right
# before it (Department).
_FACULTY_FIELD_LABELS = [
    "Name",
    "Title",
    "Faculty",
    "Department",
    "Contact",
    "Email",
    "Phone",
    "Office",
    "Biography",
    "Research Interests",
    "Website",
    "Office Hours",
]

_FACULTY_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _FACULTY_FIELD_LABELS
)

# Fields shown on the card, in display order. Name/Title/Department/
# Email/Office/Phone/Research Interests are Requirement 3's named
# fields; Faculty and Biography are additional - both appear in the
# existing raw context and dropping them would be real information
# loss (the same mistake Phase 13D found and fixed for the Course
# Card). Website/Office Hours have no corresponding column in
# faculty.db at all (confirmed directly against the schema) - they're
# still listed so parsing would pick them up if that data ever exists,
# but today they'll always be absent and simply hidden, per
# Requirement 3's "hide missing fields."
_FACULTY_CARD_FIELDS = [
    "Name",
    "Title",
    "Faculty",
    "Department",
    "Email",
    "Office",
    "Phone",
    "Research Interests",
    "Biography",
    "Website",
    "Office Hours",
]


def _parse_faculty_fields(answer):
    """Mirrors _parse_course_fields() - best-effort parse of the existing
    label:value faculty text. Returns None (parsing "failed") unless Name
    is found - the one field that actually identifies a person - since
    "faculty_profile" answer text can still be LLM-paraphrased prose in
    older stored session history from before this phase, with no labels
    to find at all. render_faculty() falls back to the original
    rendering whenever this returns None."""

    fields = {
        label: _extract_labeled_field(label, answer, _FACULTY_FIELD_ALTERNATION)
        for label in _FACULTY_CARD_FIELDS
    }

    if not fields["Name"]:
        return None

    return fields


def _render_faculty_fallback(answer, source):

    st.markdown(f"👨‍🏫 Faculty\n\n{answer}")
    _render_source(source)


# A single st.markdown() call, not one per field - same reasoning as
# render_course() (the evaluate.py AppTest harness reads only the first
# markdown element of the last chat message).
def render_faculty(answer, source):

    fields = _parse_faculty_fields(answer)

    if not fields:
        _render_faculty_fallback(answer, source)
        return

    lines = ["👨‍🏫 Faculty", ""]

    for label in _FACULTY_CARD_FIELDS:

        value = fields.get(label)

        if value:
            lines.append(f"**{label}**")
            lines.append(value)
            lines.append("")

    st.markdown("\n".join(lines).rstrip())
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
