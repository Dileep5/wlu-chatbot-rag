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


# Header + answer are a single st.markdown() call (not two) so the
# evaluate.py AppTest harness - which reads only the first markdown
# element of the last chat message - still sees the full answer text,
# unchanged from before this refactor.

def render_course(answer, source):

    st.markdown(f"📘 Course\n\n{answer}")
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
