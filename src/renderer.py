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


# Phase 13F: known labels the raw "program" context can contain
# (search_program()'s own f-string in retriever.py) - same lookahead-
# boundary pattern as course/faculty parsing.
_PROGRAM_FIELD_LABELS = [
    "Program",
    "Level",
    "Program Type",
    "Description",
    "Admission Requirements",
    "Program Requirements",
]

_PROGRAM_FIELD_ALTERNATION = "|".join(
    re.escape(label) for label in _PROGRAM_FIELD_LABELS
)

# (display label, raw label) pairs, in display order. Program Name/
# Level/Admission Requirements/Description are Requirement 3's named
# fields the raw text actually has a label for ("Program" is renamed to
# "Program Name" to match Requirement 3's wording). Degree/Department/
# Required Courses/Electives/Credits Required/Coordinator/Duration have
# no corresponding label anywhere in this response_type's text (this is
# a real data gap, not a parsing miss - confirmed directly against
# retriever.py's PROGRAM branch, which never produces any of them), so
# they're simply never found and hidden, per Requirement 3's "hide
# missing fields." Program Type/Program Requirements are additional
# labels the raw text does contain - shown per Requirement 4 rather
# than dropped, the same lesson already applied to the Course and
# Faculty cards.
_PROGRAM_CARD_FIELD_MAP = [
    ("Program Name", "Program"),
    ("Level", "Level"),
    ("Program Type", "Program Type"),
    ("Admission Requirements", "Admission Requirements"),
    ("Program Requirements", "Program Requirements"),
    ("Description", "Description"),
]


def _parse_program_fields(answer):
    """Mirrors _parse_course_fields()/_parse_faculty_fields() - returns
    None (parsing "failed") unless Program Name is found, since that's
    the one field that actually identifies a program. Text for
    "undergraduate_requirements"/"graduate_requirements"/
    "undergraduate_program_list" response_types never has this label at
    all (they're sentences/bulleted lists, not a label:value program
    sheet), so this correctly returns None for those and render_program()
    falls back to the original rendering - unchanged from before this
    phase for all three."""

    fields = {
        display: _extract_labeled_field(raw, answer, _PROGRAM_FIELD_ALTERNATION)
        for display, raw in _PROGRAM_CARD_FIELD_MAP
    }

    if not fields["Program Name"]:
        return None

    return fields


def _render_program_fallback(answer, source):

    st.markdown(f"🎓 Program\n\n{answer}")
    _render_source(source)


# Phase 13F: known labels the raw "program" context can contain
# Phase 14B: real scraped undergraduate program descriptions (confirmed
# directly against live data, e.g. "Honours BSc Computer Science") embed
# a year-by-year course breakdown and a regulations block as literal,
# standalone-line headings within the Description text itself - "Year
# 1"/"Year 2"/"Year 3"/"Year 4" and "Program Regulations". These are
# used as section boundaries below; nothing here changes what
# retriever.py produces, only how this already-parsed text gets
# reorganized for display.
_YEAR_HEADING_PATTERN = re.compile(r"(?m)^[ \t]*Year[ \t]+([1-4])[ \t]*$")

_PROGRAM_REGULATIONS_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*Program Regulations[ \t]*$"
)

# Graduate program descriptions (confirmed directly, e.g. "Master of
# Applied Politics") use a different standalone-line heading instead of
# "Program Regulations" - "Additional Information", immediately followed
# by a short page-navigation-style list of the sub-headings that repeat
# right after it (which duplicate the separately-labeled Admission
# Requirements/Program Requirements text already captured on their own).
# Recognizing it as a boundary too keeps that duplicate tail out of
# Overview, the same way "Program Regulations" already does for
# undergraduate descriptions.
_ADDITIONAL_INFO_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*Additional Information[ \t]*$"
)

# A confirmed, recurring scraped-page footer artifact (seen verbatim in
# multiple real program descriptions) - not a real "regulations" fact,
# but not discarded either: routed into Additional Information's "Other
# notes" rather than left mixed into Program Regulations.
_TRAILING_BOILERPLATE_PATTERN = re.compile(
    r"Academic\s*&\s*Related\s*Dates", re.IGNORECASE
)

# Sentence-level keywords used only to decide which sentence, within the
# text following "Program Regulations", belongs under Additional
# Information (campus/delivery mode/co-op) rather than staying under
# Program Regulations proper (GPA/graduation/progression). This is a
# best-effort classification, not exact NLP - any sentence that matches
# none of these simply stays under Program Regulations, so nothing is
# ever dropped regardless of how well a given sentence classifies.
_ADDITIONAL_INFO_KEYWORDS = [
    "campus", "campuses",
    "delivery", "in-person", "hyflex", "hybrid", "online", "remote",
    "co-op", "cooperative education", "internship", "placement",
]


def _split_regulations_and_additional(text):
    """Splits the text following the "Program Regulations" heading into
    (regulations, additional) - sentences matching a campus/delivery/
    co-op keyword go to "additional", the trailing scraped-page footer
    (if present) always goes to "additional" as a catch-all "other
    notes", and everything else stays in "regulations". Never drops
    anything: every sentence ends up in exactly one of the two."""

    if not text:
        return None, None

    boilerplate_match = _TRAILING_BOILERPLATE_PATTERN.search(text)

    if boilerplate_match:
        main_text = text[:boilerplate_match.start()].strip()
        footer_text = text[boilerplate_match.start():].strip()
    else:
        main_text = text
        footer_text = None

    chunks = re.split(r"(?<=[.!?])\s+", main_text) if main_text else []

    regulation_chunks = []
    additional_chunks = []

    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        lowered = chunk.lower()

        if any(keyword in lowered for keyword in _ADDITIONAL_INFO_KEYWORDS):
            additional_chunks.append(chunk)
        else:
            regulation_chunks.append(chunk)

    if footer_text:
        additional_chunks.append(footer_text)

    regulations = " ".join(regulation_chunks).strip() or None
    additional = "\n\n".join(additional_chunks).strip() or None

    return regulations, additional


def _split_program_description(description):
    """Best-effort split of the Description text into an overview,
    a year-by-year schedule, a regulations block, and an additional-
    information block - every one of which may be empty/None if this
    particular description doesn't contain that structure at all (e.g.
    a short graduate description with no year breakdown), in which case
    the whole thing simply stays as "overview". Every character of the
    input ends up in exactly one of the four returned pieces - nothing
    is ever discarded, only reorganized."""

    if not description:
        return {"overview": None, "schedule": [], "regulations": None, "additional": None}

    regulations_match = _PROGRAM_REGULATIONS_HEADING_PATTERN.search(description)
    additional_match = _ADDITIONAL_INFO_HEADING_PATTERN.search(description)

    # Whichever of the two non-year headings appears first - used both
    # to bound the schedule below and as an Overview boundary alongside
    # the year headings.
    non_year_positions = [
        m.start() for m in (regulations_match, additional_match) if m
    ]
    earliest_non_year = min(non_year_positions) if non_year_positions else None

    year_matches = list(_YEAR_HEADING_PATTERN.finditer(description))

    # Only years found BEFORE that boundary count as schedule boundaries -
    # real data never puts them after, but this keeps the boundary math
    # below unambiguous if it ever did.
    if earliest_non_year is not None:
        year_matches = [m for m in year_matches if m.start() < earliest_non_year]

    boundary_positions = [m.start() for m in year_matches]

    if earliest_non_year is not None:
        boundary_positions.append(earliest_non_year)

    first_boundary = min(boundary_positions) if boundary_positions else None

    overview = (
        description[:first_boundary].strip()
        if first_boundary is not None
        else description.strip()
    )

    schedule = []

    if year_matches:

        next_boundaries = [m.start() for m in year_matches[1:]]
        next_boundaries.append(
            earliest_non_year if earliest_non_year is not None else len(description)
        )

        for match, boundary_end in zip(year_matches, next_boundaries):

            year_text = description[match.end():boundary_end].strip()

            if year_text:
                schedule.append((f"Year {match.group(1)}", year_text))

    regulations_text = None
    additional_chunks = []

    if regulations_match:

        # Bounded by the Additional Information heading if one follows
        # it, otherwise runs to the end of the description - either way,
        # keyword-based sentence classification is always applied, so a
        # campus/delivery/co-op mention (or the trailing page footer)
        # still surfaces under Additional Information rather than
        # staying mixed into genuine GPA/graduation/progression text,
        # regardless of which heading combination this description has.
        regulations_end = (
            additional_match.start()
            if additional_match and additional_match.start() > regulations_match.start()
            else len(description)
        )
        remainder = description[regulations_match.end():regulations_end].strip()
        regulations_text, extra_additional = _split_regulations_and_additional(remainder)

        if extra_additional:
            additional_chunks.append(extra_additional)

    if additional_match:

        additional_remainder = description[additional_match.end():].strip()

        if additional_remainder:
            additional_chunks.append(additional_remainder)

    additional_text = "\n\n".join(additional_chunks).strip() or None

    return {
        "overview": overview or None,
        "schedule": schedule,
        "regulations": regulations_text,
        "additional": additional_text,
    }


def render_program(answer, source):

    fields = _parse_program_fields(answer)

    if not fields:
        _render_program_fallback(answer, source)
        return

    try:
        sections = _split_program_description(fields.get("Description"))
    except Exception:
        _render_program_fallback(answer, source)
        return

    lines = ["🎓 Program", ""]

    # 🎓 Program Information - Program Name/Level/Program Type are the
    # only fields of Requirement 3's list this response_type's text ever
    # has (Degree/Department have no corresponding label anywhere in it -
    # a real data gap, not a parsing miss, confirmed in Phase 13F).
    info_fields = [
        ("Program Name", fields.get("Program Name")),
        ("Level", fields.get("Level")),
        ("Program Type", fields.get("Program Type")),
    ]

    if any(value for _, value in info_fields):

        lines.append("## 🎓 Program Information")
        lines.append("")

        for label, value in info_fields:

            if value:
                lines.append(f"**{label}**")
                lines.append(value)
                lines.append("")

    # 📥 Admission Requirements - not in Requirement 3's named section
    # list, but the raw text does carry it as its own label (graduate
    # programs) - preserved as its own section rather than dropped, the
    # same "don't lose extra labeled fields" precedent already applied
    # to the Course and Faculty cards.
    if fields.get("Admission Requirements"):

        lines.append("## 📥 Admission Requirements")
        lines.append("")
        lines.append(fields["Admission Requirements"])
        lines.append("")

    # 📝 Overview
    if sections["overview"]:

        lines.append("## 📝 Overview")
        lines.append("")
        lines.append(sections["overview"])
        lines.append("")

    # 📚 Required Courses - graduate programs describe these in prose
    # under the "Program Requirements" label (already parsed above);
    # undergraduate programs organized by year instead list them inside
    # each Year's own block below, so this section is left hidden for
    # those rather than duplicating the same courses under two headings.
    if fields.get("Program Requirements"):

        lines.append("## 📚 Required Courses")
        lines.append("")
        lines.append(fields["Program Requirements"])
        lines.append("")

    # 📅 Recommended Schedule
    if sections["schedule"]:

        lines.append("## 📅 Recommended Schedule")
        lines.append("")

        for year_label, year_text in sections["schedule"]:
            lines.append(f"**{year_label}**")
            lines.append(year_text)
            lines.append("")

    # 📋 Program Regulations
    if sections["regulations"]:

        lines.append("## 📋 Program Regulations")
        lines.append("")
        lines.append(sections["regulations"])
        lines.append("")

    # ℹ️ Additional Information
    if sections["additional"]:

        lines.append("## ℹ️ Additional Information")
        lines.append("")
        lines.append(sections["additional"])
        lines.append("")

    # 🔗 Source - the actual link is still rendered by the shared
    # _render_source() helper (unchanged, still used identically by the
    # Course/Faculty/generic renderers) in its own call right after this
    # one; this heading just labels it consistently with the sections
    # above, inside the same combined block the AppTest harness reads.
    if source:
        lines.append("## 🔗 Source")
        lines.append("")

    st.markdown("\n".join(lines).rstrip())
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
