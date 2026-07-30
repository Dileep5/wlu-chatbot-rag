import os
import re

import streamlit as st
from openai import OpenAI

from retriever import (
    hybrid_search,
    structured_search,
    resolve_contextual_reference,
    FOLLOWUP_PHRASES,
    normalize_followup_text,
    create_memory,
)
from conversation import is_conversation
from domain_guard import is_wlu_related
from renderer import render_response
import citation

# -----------------------------
# Configuration
# -----------------------------

GREETING_PATTERNS = [
    r"^hi$",
    r"^hello$",
    r"^hey$",
    r"^hi bro$",
    r"^hello bro$",
    r"^hey bro$",
    r"^hii+$",
    r"^heyy+$",
    r"^yo$",
    r"^good morning$",
    r"^good afternoon$",
    r"^good evening$",
]

OFF_TOPIC_MESSAGE = (
    "I'm designed only to help with questions about "
    "Wilfrid Laurier University - things like programs, "
    "courses, admissions, tuition, faculty, campus, "
    "scholarships, student services, and departments. "
    "I'm not able to help with that topic, but feel free "
    "to ask me anything about WLU!"
)


# -----------------------------
# Helper Functions
# -----------------------------

def is_greeting(text: str) -> bool:

    text = text.lower().strip()

    for pattern in GREETING_PATTERNS:

        if re.fullmatch(pattern, text):
            return True

    return False


# Phase 12C: response_type values that structured_search()/hybrid_search()
# identify as already-complete, deterministic text (Phase 12A/12B found the
# LLM added little value re-phrasing these) - generate_answer() shows them
# directly instead of spending an LLM call to re-word what's already a
# correct, final answer. Every other response_type (department_profile/
# undergraduate_program_list/vector/etc.) is unaffected and still goes
# through the LLM exactly as before.
#
# Phase 13D: "course" added so renderer.py's Course Card (Phase 13C) has
# the raw "Course Code:"/"Course Name:"/... labeled text to parse - an
# LLM paraphrase of that text (the prior behavior) almost never
# reproduces those labels verbatim, which is why the card previously
# fell back to the plain rendering for nearly every live query.
#
# Phase 13E: "faculty_profile" added for the same reason, so the Faculty
# Card has the raw "Name:"/"Title:"/... labeled text to parse instead of
# an LLM paraphrase of it.
#
# Phase 13F: "program" added for the same reason, so the Program Card has
# the raw "Program:"/"Level:"/... labeled text to parse. "undergraduate_
# requirements"/"graduate_requirements" were already deterministic, and
# "undergraduate_program_list" is a bulleted list of many programs, not
# a single program's fields - the card can't apply to it regardless, so
# it's deliberately left on the LLM path, unchanged.
#
# "faculty_clarify" added so a multi-candidate faculty-name
# disambiguation message (structured_search's FACULTY branch,
# retriever.py) is shown verbatim - the same reason every other
# clarification message in this project bypasses the LLM (see
# resolve_contextual_reference): an LLM paraphrase risks dropping or
# inventing a candidate name.
#
# "not_found" added for the same hallucination-prevention reason: every
# message using this response_type (structured_search's course/faculty/
# program "not found" checks, and hybrid_search's low-confidence vector
# gate) is a fixed, already-correct decline with no citation - handing
# it to the LLM to "answer naturally" would risk exactly the fabrication
# this response_type exists to prevent.
#
# "course_clarify" added for the same reason as "faculty_clarify": a
# multi-candidate course-NAME disambiguation message (structured_search's
# COURSE branch, retriever.py) is shown verbatim so an LLM paraphrase
# can't drop or invent a candidate course code.
#
# Phase 18: "department_profile" added for the same reason as course/
# faculty_profile/program (Phase 13D/13E/13F) - renderer.py's new
# Department Card needs the raw "Department:"/"Faculty:"/... labeled
# text to parse; an LLM paraphrase of it (the prior behavior) has none
# of those labels, so the card would otherwise always fall back to
# plain text instead of rendering.
DETERMINISTIC_RESPONSE_TYPES = {
    "prerequisite",
    "undergraduate_requirements",
    "graduate_requirements",
    "coordinator",
    "research",
    "course",
    "faculty_profile",
    "program",
    "faculty_clarify",
    "not_found",
    "course_clarify",
    "department_profile",
}


def generate_answer(
    query,
    context,
    response_type=None
):

    if response_type in DETERMINISTIC_RESPONSE_TYPES:
        return context.strip()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        return (
            "OpenAI API key not found."
        )

    client = OpenAI()

    messages = [

        {
            "role": "system",
            "content": """
You are the official AI assistant
for Wilfrid Laurier University.

You should behave naturally and
conversationally, similar to ChatGPT
or Claude, but every specific fact
you state must come from the
retrieved WLU information you are
given below, never from your own
general knowledge.

Responsibilities:

1. Have natural conversations.
2. Help users with WLU.
3. Answer follow-up questions.
4. Remember previous messages.
5. Use retrieved WLU information
   whenever available.

Grounding rules (critical):

- Only state specific facts that are
  explicitly present in the retrieved
  WLU information below.
- Never invent or estimate specific
  details that aren't shown in the
  retrieved text - this includes
  tuition amounts, scholarship values
  or types, eligibility requirements,
  deadlines, statistics, and program
  details.
- If the retrieved information only
  partially covers the question,
  summarize what it DOES say and note
  what it doesn't cover - don't refuse
  to answer just because it's
  incomplete.
- If the retrieved information doesn't
  meaningfully address the question at
  all, say plainly that the available
  WLU data doesn't contain enough
  information to answer confidently,
  instead of answering from outside
  knowledge.

Rules:

- Be friendly.
- Be conversational.
- Format answers clearly.
- Only answer questions related to
  Wilfrid Laurier University. If asked
  about something unrelated, politely
  say you can only help with WLU topics.
"""
        }

    ]

    messages.extend(
        st.session_state.chat_history[-10:]
    )

    messages.append(
        {
            "role": "user",
            "content": f"""
Retrieved WLU Information:

{context}

Question:

{query}

Answer using only the facts stated
above. Do not add specific figures,
names, dates, or requirements that
aren't explicitly present in the
retrieved information.
"""
        }
    )

    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return answer


def generate_chat_response(query):

    client = OpenAI()

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",
                "content":
                """
You are a friendly AI assistant
for Wilfrid Laurier University.

Behave naturally like ChatGPT
or Claude.

You may have natural
conversations with users.

Examples:

- how are you
- who are you
- what can you do
- do you know me
- tell me a joke
- what did i ask before

Be friendly and conversational.
                """
            }

        ]

        +

        st.session_state.chat_history[-10:]

        +

        [
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0.8,
        max_tokens=300
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# -----------------------------
# Custom CSS (Phase 17: visual redesign only - no widget behavior,
# routing, memory, prompting, or rendering logic below this point is
# touched; every st.markdown(..., unsafe_allow_html=True) call here
# either injects a page-wide <style> block or purely decorative HTML
# that replaces an equivalent plain st.title()/st.caption()/st.info()
# call one-for-one).
# -----------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --wlu-purple: #4B2E83;
    --wlu-purple-dark: #34215C;
    --wlu-purple-soft: #F2EEFA;
    --wlu-gold: #C9A227;
    --wlu-ink: #201C2E;
    --wlu-ink-muted: #675F7D;
    --wlu-border: #E4DFF0;
    --wlu-online: #1F9D55;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 880px;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', 'Inter', sans-serif;
    letter-spacing: -0.01em;
}

/* Hero */
.wlu-hero {
    background: linear-gradient(135deg, var(--wlu-purple) 0%, var(--wlu-purple-dark) 100%);
    border-radius: 18px;
    padding: 2.25rem 2.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 28px rgba(75, 46, 131, 0.22);
}
.wlu-hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.4rem;
}
.wlu-hero .wlu-tagline {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--wlu-gold);
    margin: 0 0 0.65rem;
}
.wlu-hero .wlu-desc {
    font-size: 0.94rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.86);
    max-width: 62ch;
    margin: 0;
}

/* Welcome card */
.wlu-welcome {
    background: var(--wlu-purple-soft);
    border: 1px solid var(--wlu-border);
    border-left: 4px solid var(--wlu-purple);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
.wlu-welcome .wlu-welcome-title {
    font-weight: 700;
    color: var(--wlu-purple-dark);
    font-size: 0.98rem;
}
.wlu-welcome p {
    margin: 0.4rem 0 0;
    color: var(--wlu-ink-muted);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Sidebar */
section[data-testid="stSidebar"] .wlu-side-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--wlu-purple-dark);
    margin-bottom: 1rem;
}
.wlu-status-card {
    background: #FFFFFF;
    border: 1px solid var(--wlu-border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}
.wlu-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.32rem 0;
    font-size: 0.85rem;
}
.wlu-status-row + .wlu-status-row {
    border-top: 1px solid var(--wlu-border);
}
.wlu-status-row .wlu-label {
    color: var(--wlu-ink-muted);
    display: flex;
    align-items: center;
}
.wlu-status-row .wlu-value {
    font-weight: 600;
    color: var(--wlu-ink);
}
.wlu-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--wlu-online);
    margin-right: 7px;
    box-shadow: 0 0 0 3px rgba(31, 157, 85, 0.15);
}
.wlu-version {
    text-align: center;
    font-size: 0.76rem;
    color: var(--wlu-ink-muted);
    margin-top: 1.25rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--wlu-border);
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    border: 1px solid var(--wlu-border);
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(32, 28, 46, 0.04);
}

/* Suggested-question buttons */
.stButton > button {
    border-radius: 999px !important;
    border: 1px solid var(--wlu-border) !important;
    font-size: 0.86rem !important;
    color: var(--wlu-ink) !important;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--wlu-purple) !important;
    color: var(--wlu-purple) !important;
}
</style>
"""


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="WLU Hybrid RAG Assistant",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="wlu-hero">
        <h1>WLU Hybrid RAG Assistant</h1>
        <p class="wlu-tagline">Grounded AI Assistant for Wilfrid Laurier University</p>
        <p class="wlu-desc">
            Hybrid RAG combines deterministic structured retrieval over
            real WLU data with vector search and an LLM, so every answer
            about courses, programs, faculty, admissions, tuition, and
            student services is grounded in scraped WLU records - never
            general knowledge.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="wlu-welcome">
        <span class="wlu-welcome-title">👋 Welcome!</span>
        <p>
            I'm a hybrid RAG assistant for Wilfrid Laurier University.
            I can help with course details, program and admission
            requirements, faculty profiles, scholarships, tuition, and
            student services - grounded in real WLU data, not general
            knowledge.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.markdown(
        '<div class="wlu-side-title">🎓 WLU Hybrid RAG Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="wlu-status-card">
            <div class="wlu-status-row">
                <span class="wlu-label"><span class="wlu-dot"></span>Status</span>
                <span class="wlu-value">Online</span>
            </div>
            <div class="wlu-status-row">
                <span class="wlu-label">Model</span>
                <span class="wlu-value">GPT-4o-mini</span>
            </div>
            <div class="wlu-status-row">
                <span class="wlu-label">Retriever</span>
                <span class="wlu-value">Hybrid RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📚 Knowledge Sources")
    st.markdown(
        "- Academic Calendar\n"
        "- Faculty Directory\n"
        "- WLU Website"
    )

    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown(
        "This assistant pairs deterministic structured retrieval "
        "(courses, programs, faculty) with vector search and an "
        "LLM, grounding every answer in real, scraped Wilfrid "
        "Laurier University data rather than general knowledge."
    )

    st.markdown(
        '<div class="wlu-version">Version 1.0.0</div>',
        unsafe_allow_html=True
    )


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = create_memory()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------
# Suggested Questions - shown only before the first turn, and hidden on
# the one rerun immediately after a suggestion is clicked (pending_query
# set) so they never overlap with the chat reply that's about to render.
# -----------------------------

SUGGESTED_QUESTIONS = [
    "What is CP312?",
    "Tell me about CP317.",
    "Tell me about Ammara Mahmood.",
    "Tell me about the Honours BSc Computer Science program.",
    "What scholarships are available?",
    "What are the admission requirements?",
]

st.divider()

show_suggestions = (
    not st.session_state.messages
    and not st.session_state.pending_query
)

if show_suggestions:

    st.markdown("#### 💡 Try asking:")

    cols = st.columns(2)

    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):

        if cols[i % 2].button(
            suggestion,
            key=f"suggested_question_{i}",
            use_container_width=True
        ):
            st.session_state.pending_query = suggestion
            st.rerun()

    st.divider()


# -----------------------------
# Display Previous Messages
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        render_response(
            msg.get("response_type"),
            msg["content"],
            msg.get("source"),
        )


# -----------------------------
# User Input
# -----------------------------

query = st.chat_input(
    "Ask something about WLU..."
)

if not query and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None


if query:

    # One increment per user turn, read by every retrieval write-back
    # site as the entity-history "turn_number" stamp (Sprint 9B) - done
    # here, once, rather than in retriever.py, so a single turn that
    # calls structured_search multiple times (e.g. resolve_contextual_
    # reference re-invoking it with a rewritten question) still counts
    # as one turn.
    st.session_state.memory["turn_count"] = (
        st.session_state.memory.get("turn_count", 0) + 1
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            query
        )

    try:

        # greetings
        if is_greeting(query):

            answer = (
                "Hello! 👋\n\n"
                "How can I help you "
                "with Wilfrid Laurier "
                "University today?"
            )

            source = None
            response_type = None

        # normal conversation
        elif is_conversation(query):

            answer = generate_chat_response(
                query
            )

            source = None
            response_type = None

        # grounded structured match (course/program/department) - a real
        # match is proof the query is in-domain, so it bypasses the
        # out-of-domain gate entirely. This matters for short queries like
        # "What is BBA?" that the keyword/LLM domain check can't reliably
        # recognize on their own, but that retrieval can resolve directly.
        elif (
            structured := structured_search(
                query,
                st.session_state.memory
            )
        ):

            context, source, response_type = structured

            answer = (
                generate_answer(
                    query,
                    context,
                    response_type
                )
            )

        # unresolved contextual reference ("it", "that professor", "the
        # second one", ...) - checked only once structured_search has
        # already failed on the raw question. A real match is grounded
        # exactly like the structured branch above; a failure to resolve
        # returns a clarification directly and must never fall through
        # to hybrid/vector search, which is what let these produce
        # confident, fabricated answers before (Sprint 7A).
        elif (
            contextual := resolve_contextual_reference(
                query,
                st.session_state.memory
            )
        ):

            if contextual[0] == "resolved":

                _, context, source, response_type = contextual

                answer = (
                    generate_answer(
                        query,
                        context,
                        response_type
                    )
                )

            else:

                answer = contextual[1]
                source = None
                response_type = None

        # out-of-domain (memory follow-ups always bypass this check)
        elif (
            normalize_followup_text(query) not in FOLLOWUP_PHRASES
            and not is_wlu_related(query)
        ):

            answer = OFF_TOPIC_MESSAGE
            source = None
            response_type = None

        # WLU retrieval
        else:

            context, source, response_type = (
                hybrid_search(
                    query,
                    st.session_state.memory
                )
            )

            answer = (
                generate_answer(
                    query,
                    context,
                    response_type
                )
            )

        # Phase 3: enriches whatever `source` each branch above already
        # decided on (a bare URL string, or None for non-factual
        # replies) into {"date", "sources": [{"title", "url"}, ...]} -
        # purely a presentation step, run after every retrieval/routing
        # decision above is already final. Stored in this enriched form
        # so the history-replay loop below re-renders it identically
        # without repeating any lookup.
        source = citation.build_citation(source, response_type)

        with st.chat_message(
            "assistant"
        ):

            render_response(
                response_type,
                answer,
                source
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source,
                "response_type": response_type
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        # Production polish: the user sees a friendly, generic message -
        # raw exception text (e.g. "'NoneType' object has no attribute
        # ...") is confusing and leaks implementation detail. The real
        # exception is still printed to the console for whoever's
        # running the app, unchanged in substance from before.
        print(f"Unhandled error while answering {query!r}: {e}")

        error_msg = (
            "Sorry, something went wrong while answering that. "
            "Please try rephrasing your question, or ask again in a "
            "moment."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_msg
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_msg
            )