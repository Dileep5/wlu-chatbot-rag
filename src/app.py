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
#
# "course_instructors"/"faculty_list"/"department_faculty_list" added
# here (natural-language push): these three were the only structured,
# already-complete list-shaped answers (built by retriever.py's own
# _format_faculty_list_context()) that were STILL being routed through
# the LLM paraphrase path despite every sibling list type ("research")
# already being deterministic - an oversight, not a deliberate
# distinction, confirmed by checking retriever.py directly (their
# context text has exactly the same "- Name (Title)" bullet-list shape
# "research" already gets verbatim). Fixing this is required for the
# natural-language lead-in below to mean what it says: a summary ABOVE
# the real bulleted list, not above an LLM's own re-paraphrase of it.
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
    "course_instructors",
    "faculty_list",
    "department_faculty_list",
    # Phase 2: policy-index lookups (search_policy, retriever.py) are
    # already-complete "Policy N.N: Title" text with no labeled fields
    # for a dedicated card to parse - shown verbatim for the same
    # hallucination-prevention reason as every other type above, not
    # paraphrased by the LLM.
    "policy",
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
- Only answer questions related to
  Wilfrid Laurier University. If asked
  about something unrelated, politely
  say you can only help with WLU topics.

Formatting (visual structure, not content -
every rule above about what you can say
still applies exactly as written):

- Use markdown to make the answer easy to
  scan, the way a well-formatted web answer
  looks - not a single dense paragraph.
- **Bold** key terms, names, numbers, and
  requirements the user is likely scanning
  for (deadlines, course codes, amounts,
  names).
- When the answer covers multiple items,
  steps, or requirements, use a short
  bulleted or numbered list instead of
  running them together in prose.
- Keep paragraphs short - a few sentences
  at most before a line break, list, or new
  paragraph.
- Formatting must never change or hide a
  fact - it's purely how the same grounded
  content is laid out.

Conversational follow-ups (occasional, situational):

- Sometimes, when it naturally fits the
  topic, end your answer with one short,
  genuine follow-up question about the
  user's specific situation - the way a
  curious person would, not a canned
  prompt, and not on every answer.
- Only ask this if the retrieved
  information above actually distinguishes
  different cases relevant to the question
  (e.g. domestic vs. international
  eligibility, undergraduate vs. graduate,
  different campuses or terms). For
  example, after answering about
  scholarships that differ by student
  status, you could ask "Are you a domestic
  or international student? That affects
  which ones you'd qualify for."
- Never ask a follow-up question that
  implies information exists if the
  retrieved text above doesn't actually
  contain it - only ask about a distinction
  that's genuinely present in the retrieved
  information.
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


# Response types that get a natural-language lead-in generated by
# generate_grounded_summary() below, shown ABOVE whatever verbatim/card
# content already renders beneath it - never a replacement for that
# content, just a plain-English framing sentence on top of it.
#
# The original four (course/faculty_profile/program/department_profile)
# are the full labeled "spec sheet" cards. The nine added afterward
# (prerequisite/coordinator/undergraduate_requirements/
# graduate_requirements/policy/research/faculty_list/
# department_faculty_list/course_instructors) are shorter, already-
# terse deterministic answers (a single fact, a bulleted list, a policy
# entry) - terse enough on their own that a short lead-in sentence
# meaningfully softens them the same way it did for the four cards,
# without ever replacing the underlying list/fact text they're built
# from. Every member of this set must also be in
# DETERMINISTIC_RESPONSE_TYPES above: the summary is only ever grounded
# in the exact same verbatim context that gets shown below it, never in
# an LLM's own prior paraphrase of that context.
SUMMARY_RESPONSE_TYPES = {
    "course",
    "faculty_profile",
    "program",
    "department_profile",
    "prerequisite",
    "coordinator",
    "undergraduate_requirements",
    "graduate_requirements",
    "policy",
    "research",
    "faculty_list",
    "department_faculty_list",
    "course_instructors",
}


def generate_grounded_summary(query, context, response_type):
    """A 1-2 sentence natural-language lead-in for every response_type
    listed in SUMMARY_RESPONSE_TYPES above, generated ONLY from
    `context` - the exact same already-deterministic text (e.g. "Course
    Code: ...\\nCourse Name: ..." or a "Faculty members:\\n- ..." bullet
    list) that generate_answer() returns verbatim for these response
    types via DETERMINISTIC_RESPONSE_TYPES, never raw scraped page text
    and never the LLM's own outside knowledge. Deliberately kept
    structurally separate from that context string - the caller must
    thread the return value through as its own value (a "summary" key
    alongside "content"/"source"/"response_type"), never concatenated
    into `context`/`answer` itself, since that string is what
    renderer.py's _extract_labeled_field() parses for the card's actual
    fields, and mixing free-form prose into it risks exactly the kind
    of mis-parse this project has already hit more than once this
    session.

    Returns None (no summary shown) whenever response_type isn't in
    SUMMARY_RESPONSE_TYPES, or the API key is missing, or the call
    itself fails for any reason - a missing summary just means the
    response renders exactly as it did before this feature existed,
    never a broken response."""

    if response_type not in SUMMARY_RESPONSE_TYPES:
        return None

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Using ONLY the facts listed below, answer the "
                        "user's question in 1-2 sentences the way a "
                        "genuinely curious, warm person would - "
                        "conversational, with real personality, never a "
                        "dry textbook restatement. Do not add, infer, or "
                        "estimate any name, number, date, or requirement "
                        "that isn't explicitly present below. If the "
                        "listed facts don't fully answer the question, "
                        "say what they do cover and note what's missing "
                        "- never fill the gap from outside knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": f"Facts:\n\n{context}\n\nQuestion:\n\n{query}"
                }
            ],
            temperature=0.25,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception:

        return None


# Bounded, capability-real follow-up hints shown under certain
# response_types (Google-AI-Mode-style "Would you like to know X?").
# Deliberately a fixed dict, never LLM-generated: an LLM asked to
# "suggest a natural follow-up" has no way to know which follow-up
# phrasings this app's own deterministic routing (resolve_contextual_
# reference/structured_search, retriever.py) can actually resolve, so a
# free-generated suggestion risks promising a capability that doesn't
# exist - exactly the failure mode this dict exists to avoid.
#
# Every phrasing below was verified LIVE against the real routing
# (structured_search() to establish the entity in memory, then
# resolve_contextual_reference() on the suggested follow-up text
# itself) before being added - not assumed from the response_type's
# name or from what "should" work:
#   - "prerequisites"/"who teaches it" resolve via _INTENT_REWRITE_
#     RULES's course-typed rewrite rules.
#   - "who coordinates it/their department" resolves via _COORDINATOR_
#     REWRITE_PATTERN/_attempt_coordinator_resolution against whichever
#     of program/department/(a faculty member's own department) was
#     most recently established.
#   - "tell me about the first one" resolves via _resolve_ordinal_
#     entity() against memory["_last_list_id"], written by every list-
#     shaped structured branch (_record_entity_list()) - confirmed
#     directly for course_instructors/research and, since faculty_list/
#     department_faculty_list share that exact same recording call,
#     true for them too.
# A response_type with no verified-safe follow-up (coordinator, policy,
# and anything not listed) is simply absent - no line is shown, rather
# than guessing at one that might not resolve.
FOLLOWUP_SUGGESTIONS = {
    "course": "Curious about the prerequisites, or who's teaching it this year?",
    "prerequisite": "Want to know who's teaching it these days?",
    "course_instructors": (
        'Curious what you need to take first, or want to know more about '
        'one of them? Try "Tell me about the first one."'
    ),
    "faculty_profile": "Curious who coordinates their department?",
    "program": "Want to know who's coordinating this program?",
    "undergraduate_requirements": "Curious who coordinates this program?",
    "graduate_requirements": "Curious who coordinates this program?",
    "department_profile": "Want to know who coordinates this department?",
    "faculty_list": (
        'Any of them catch your eye? Try "Tell me about the first one" '
        'and I\'ll dig in.'
    ),
    "department_faculty_list": (
        'Want the scoop on one of them? Try "Tell me about the first one."'
    ),
    "research": (
        'Want to hear more about any of them? Try "Tell me about the '
        'first one" and I\'ll fill you in.'
    ),
}


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
    /* Verified against Wilfrid Laurier University's own live site CSS
       (wlu.ca's production stylesheet) - --wlu-purple and --wlu-gold
       are WLU's actual brand colors (#330072 is their site-wide link/
       brand purple; #F2A900 appears in their CSS under a class
       literally named ".Gold"), not an approximation. --wlu-purple-
       accent is their real hover/highlight purple (#924DA7 - the
       "mauve" WLU's own marketing materials pair with purple). The
       -dark/-soft/border shades are derived tints, not independently
       verified, since WLU's site doesn't need them for this component
       system's specific roles (hero gradient depth, card backgrounds). */
    --wlu-purple: #330072;
    --wlu-purple-dark: #220050;
    --wlu-purple-accent: #924DA7;
    --wlu-purple-soft: #EFEBF4;
    --wlu-gold: #F2A900;
    --wlu-ink: #201C2E;
    --wlu-ink-muted: #675F7D;
    --wlu-border: #E3DCEC;
    --wlu-online: #1F9D55;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, "Segoe UI", sans-serif;
}

@keyframes wluFadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Single large maple-leaf silhouette (an original shape, not WLU's
   registered wordmark/logo), cropped into the corner as one bold
   watermark - matching the scale and placement WLU uses on its own
   branded background templates, rather than a tiled repeating print. */
.stApp {
    background-color: var(--wlu-purple-soft);
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='100'%20height='115'%20viewBox='0%200%20100%20115'%3E%3Cg%20fill='%23330072'%20opacity='0.1'%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20scale(1.15)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(48)%20scale(0.92)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(-48)%20scale(0.92)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(102)%20scale(0.68)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(-102)%20scale(0.68)'/%3E%3Cpath%20d='M45,62%20L55,62%20L51,108%20L49,108%20Z'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: 120% -160px;
    background-size: 1350px auto;
    background-attachment: fixed;
}

header[data-testid="stHeader"] {
    background: transparent;
    position: relative;
}
header[data-testid="stHeader"]::after {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--wlu-purple) 0%, var(--wlu-gold) 50%, var(--wlu-purple-accent) 100%);
}

/* Slim brand scrollbar, all scroll containers */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(146, 77, 167, 0.45);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--wlu-purple);
}

.block-container {
    padding-top: 2rem;
    max-width: 880px;
}

h1, h2, h3, h4 {
    font-family: 'Poppins', 'Inter', sans-serif;
    letter-spacing: -0.01em;
}
.block-container h4 {
    color: var(--wlu-purple-dark);
    font-size: 1.05rem;
    margin: 0.25rem 0 0.9rem;
}
.block-container hr {
    border-color: var(--wlu-border);
    margin: 1.75rem 0;
}

/* Hero */
.wlu-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--wlu-purple) 0%, var(--wlu-purple-dark) 100%);
    border-radius: 20px;
    padding: 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 40px rgba(51, 0, 114, 0.32), 0 2px 8px rgba(51, 0, 114, 0.18);
    animation: wluFadeUp 0.5s ease both;
}
.wlu-hero::before {
    content: '';
    position: absolute;
    top: -70px;
    right: -70px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(242, 169, 0, 0.32) 0%, rgba(242, 169, 0, 0) 70%);
    pointer-events: none;
}
.wlu-hero::after {
    content: '';
    position: absolute;
    right: -70px;
    bottom: -110px;
    width: 320px;
    height: 368px;
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='100'%20height='115'%20viewBox='0%200%20100%20115'%3E%3Cg%20fill='%23FFFFFF'%20opacity='0.07'%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20scale(1.15)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(48)%20scale(0.92)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(-48)%20scale(0.92)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(102)%20scale(0.68)'/%3E%3Cpath%20d='M-12,0%20C-12,-22%20-5,-33%200,-40%20C5,-33%2012,-22%2012,0%20C7,5%20-7,5%20-12,0%20Z'%20transform='translate(50,58)%20rotate(-102)%20scale(0.68)'/%3E%3Cpath%20d='M45,62%20L55,62%20L51,108%20L49,108%20Z'/%3E%3C/g%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
    pointer-events: none;
}
.wlu-hero-badge {
    position: absolute;
    top: 1.6rem;
    right: 1.75rem;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    border: 2px solid var(--wlu-gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 0 4px rgba(242, 169, 0, 0.14), 0 4px 14px rgba(0, 0, 0, 0.28);
}
.wlu-hero h1 {
    position: relative;
    font-size: 2.15rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 4.5rem 0.4rem 0;
    text-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
}
.wlu-hero .wlu-tagline {
    position: relative;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--wlu-gold);
    margin: 0 0 0.7rem;
}
.wlu-hero .wlu-desc {
    position: relative;
    font-size: 0.94rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.86);
    max-width: 62ch;
    margin: 0;
}

/* Welcome card */
.wlu-welcome {
    background: #FFFFFF;
    border: 1px solid var(--wlu-border);
    border-left: 4px solid var(--wlu-gold);
    border-radius: 14px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(51, 0, 114, 0.08);
    animation: wluFadeUp 0.55s ease 0.08s both;
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

/* Sidebar - a clean, solid dark-purple gradient (no watermark texture),
   matching the solid left panel of WLU's own branded background
   templates: the leaf motif lives on the lighter canvas, not here.
   Card content stays on solid white "floating" panels for contrast;
   headings/body copy printed directly on the gradient are relightened
   separately below since Streamlit's default text color assumes a
   light sidebar. */
section[data-testid="stSidebar"] {
    background-image: linear-gradient(165deg, var(--wlu-purple-dark) 0%, var(--wlu-purple) 55%, var(--wlu-purple-accent) 140%);
}
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown {
    color: rgba(255, 255, 255, 0.82);
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.18);
}
section[data-testid="stSidebar"] ul {
    list-style: none;
    padding-left: 0;
    margin: 0.5rem 0 0;
}
section[data-testid="stSidebar"] li {
    position: relative;
    padding-left: 1.15rem;
    margin-bottom: 0.4rem;
}
section[data-testid="stSidebar"] li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.55em;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--wlu-gold);
}

/* Brand lockup - a horizontal wordmark + leaf-mark treatment in the
   spirit of WLU's own "LAURIER [leaf]" logo lockup, but original
   typography and an original leaf shape, not a reproduction of their
   registered logo asset. */
.wlu-brand {
    text-align: left;
    padding: 0.5rem 0 1.6rem;
    animation: wluFadeUp 0.45s ease both;
}
.wlu-brand-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.wlu-brand-word {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.65rem;
    letter-spacing: 0.01em;
    color: #FFFFFF;
    line-height: 1;
}
.wlu-brand-mark {
    width: 34px;
    height: 38px;
    flex-shrink: 0;
    fill: var(--wlu-gold);
    filter: drop-shadow(0 0 6px rgba(242, 169, 0, 0.5));
}
.wlu-brand-tagline {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
    margin-top: 0.5rem;
}
.wlu-status-card {
    position: relative;
    overflow: hidden;
    background: #FFFFFF;
    border: 1px solid var(--wlu-border);
    border-radius: 12px;
    padding: 1rem 1rem 0.85rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 26px rgba(20, 0, 46, 0.3);
    animation: wluFadeUp 0.5s ease 0.05s both;
}
.wlu-status-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--wlu-purple), var(--wlu-gold), var(--wlu-purple-accent));
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
    color: rgba(255, 255, 255, 0.55);
    margin-top: 1.25rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.18);
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    border: 1px solid var(--wlu-border);
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(32, 28, 46, 0.05);
    transition: box-shadow 0.2s ease;
}
div[data-testid="stChatMessage"]:hover {
    box-shadow: 0 8px 22px rgba(32, 28, 46, 0.1);
}
div[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, var(--wlu-purple-accent), var(--wlu-purple)) !important;
}
div[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, var(--wlu-gold), #D68F00) !important;
}

/* Chat input */
div[data-testid="stChatInput"] {
    border-radius: 18px;
    border: 1px solid var(--wlu-border) !important;
    box-shadow: 0 4px 16px rgba(51, 0, 114, 0.08);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--wlu-purple) !important;
    box-shadow: 0 8px 22px rgba(51, 0, 114, 0.16), 0 0 0 3px rgba(242, 169, 0, 0.18);
}
button[data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background: var(--wlu-purple) !important;
    color: #FFFFFF !important;
}
button[data-testid="stChatInputSubmitButton"]:not(:disabled):hover {
    background: var(--wlu-purple-dark) !important;
}

/* Suggested-question buttons */
.stButton > button {
    border-radius: 999px !important;
    border: 1px solid var(--wlu-border) !important;
    background: #FFFFFF !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: var(--wlu-ink) !important;
    padding: 0.6rem 1.1rem !important;
    box-shadow: 0 1px 3px rgba(32, 28, 46, 0.06) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, color 0.15s ease, background 0.15s ease !important;
    animation: wluFadeUp 0.4s ease both;
}
.stButton > button:hover {
    border-color: var(--wlu-purple) !important;
    color: var(--wlu-purple) !important;
    background: var(--wlu-purple-soft) !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(51, 0, 114, 0.18) !important;
}
.stButton > button:active {
    transform: translateY(0);
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
        <div class="wlu-hero-badge">🎓</div>
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
        """
        <div class="wlu-brand">
            <div class="wlu-brand-row">
                <span class="wlu-brand-word">WLU</span>
                <svg class="wlu-brand-mark" viewBox="0 0 100 115" xmlns="http://www.w3.org/2000/svg">
                    <path d="M-12,0 C-12,-22 -5,-33 0,-40 C5,-33 12,-22 12,0 C7,5 -7,5 -12,0 Z" transform="translate(50,58) scale(1.15)" />
                    <path d="M-12,0 C-12,-22 -5,-33 0,-40 C5,-33 12,-22 12,0 C7,5 -7,5 -12,0 Z" transform="translate(50,58) rotate(48) scale(0.92)" />
                    <path d="M-12,0 C-12,-22 -5,-33 0,-40 C5,-33 12,-22 12,0 C7,5 -7,5 -12,0 Z" transform="translate(50,58) rotate(-48) scale(0.92)" />
                    <path d="M-12,0 C-12,-22 -5,-33 0,-40 C5,-33 12,-22 12,0 C7,5 -7,5 -12,0 Z" transform="translate(50,58) rotate(102) scale(0.68)" />
                    <path d="M-12,0 C-12,-22 -5,-33 0,-40 C5,-33 12,-22 12,0 C7,5 -7,5 -12,0 Z" transform="translate(50,58) rotate(-102) scale(0.68)" />
                    <path d="M45,62 L55,62 L51,108 L49,108 Z" />
                </svg>
            </div>
            <div class="wlu-brand-tagline">Hybrid RAG Assistant</div>
        </div>
        """,
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
                <span class="wlu-label">🤖 Model</span>
                <span class="wlu-value">GPT-4o-mini</span>
            </div>
            <div class="wlu-status-row">
                <span class="wlu-label">🔀 Retriever</span>
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
            msg.get("summary"),
            msg.get("followup"),
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

        # Generated exactly once here, at message-creation time - never
        # in the history-replay loop above, which re-runs on every
        # Streamlit rerun and would otherwise re-call the LLM (wastefully,
        # and non-deterministically) for every past message on every
        # interaction. Stored in the message dict below and simply
        # replayed from there afterward, satisfying "only the first time
        # a card is shown" by construction rather than needing its own
        # check.
        summary = generate_grounded_summary(
            query,
            answer,
            response_type
        )

        # Same "computed once, at message-creation time" reasoning as
        # `summary` immediately above - a fixed dict lookup, not an API
        # call, but still stored on the message rather than recomputed
        # by the history-replay loop so a response_type's mapping only
        # ever needs to be looked up once per turn.
        followup = FOLLOWUP_SUGGESTIONS.get(response_type) if response_type else None

        with st.chat_message(
            "assistant"
        ):

            render_response(
                response_type,
                answer,
                source,
                summary,
                followup
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source,
                "response_type": response_type,
                "summary": summary,
                "followup": followup
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