    }  # Closing brace for message object
import os
import re
import traceback

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from retriever import (
    hybrid_search,
    structured_search,
    resolve_contextual_reference,
    FOLLOWUP_PHRASES,
    normalize_followup_text,
    create_memory,
    # Internal helpers (underscore-prefixed, deliberately imported
    # anyway) reused directly by the button-driven follow-up actions
    # below - each is already a pure, deterministic memory lookup with
    # no free-text pattern matching inside it, exactly the existing
    # logic the buttons need to call directly rather than re-implement.
    _attempt_coordinator_resolution,
    _attempt_ordinal_resolution,
    _latest_entity_of_type,
)
from conversation import is_conversation
from domain_guard import is_wlu_related, is_factual_offtopic
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

# Distinct avatars for the two chat roles - a person icon for the user,
# the same graduation cap already used on the hero badge for the
# assistant, so the two are visually distinguishable at a glance rather
# than relying on background color alone to tell them apart.
USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🎓"


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
    # Ensure closing brace

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


# The off-topic branch below (domain_guard.is_wlu_related() already
# returned False) used to show one flat canned string, OFF_TOPIC_
# MESSAGE, for every case - a genuine "what's the weather" question and
# a "hey how are you" got the identical, un-reactive reply. These two
# functions replace that with a second classification
# (domain_guard.is_factual_offtopic()) deciding which of two warmer
# paths to take. Deliberately kept SEPARATE from generate_chat_response()
# above rather than extended in place - that function is also the
# is_conversation() fast-path's generator (a fixed, already-regression-
# tested list: "hi", "thanks", "tell me a joke", ...), and adding a
# mandatory "pivot to WLU" instruction there would change its existing,
# already-verified behavior for every one of those cases too. A new,
# narrowly-scoped function avoids that risk entirely.
def generate_offtopic_social_response(query):
    """The user's message was classified as off-topic AND purely
    social/emotional (domain_guard.is_factual_offtopic() returned
    False) - e.g. "I'm bored today", "I feel like dancing". Reacts
    genuinely to what they said, then pivots to offering WLU help.
    Never states any fact about the outside world - there is nothing
    to hallucinate here as long as the model only reacts to the user's
    own stated feeling, never adds outside information of its own."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "Hey! I'm mainly here for Wilfrid Laurier University "
            "questions, but feel free to chat. Anything about WLU "
            "I can help with?"
        )

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "The user's message is purely social, emotional, or "
                    "conversational (a greeting, small talk, or a "
                    "statement about their own mood or feelings) - it is "
                    "NOT a real question and contains no request for any "
                    "fact or information.\n\n"
                    "React warmly and genuinely to what they said - a "
                    "brief, human reaction in your own words, one short "
                    "sentence. Then, in a second short sentence, pivot to "
                    "WLU - and that second sentence must make your scope "
                    "clear, in your own natural words: that you're here "
                    "specifically for Wilfrid Laurier University topics "
                    "(e.g. 'I'm all about WLU', 'I'm here for WLU "
                    "questions', 'my focus is WLU' - vary the exact "
                    "phrasing naturally, but always include that idea), "
                    "then invite a WLU question. Two short sentences "
                    "total, no more.\n\n"
                    "CRITICAL: never state any fact about the outside "
                    "world - no weather, news, trivia, or any other "
                    "real-world information, even in passing. Only react "
                    "to what the user themselves said about their own "
                    "feelings or greeting - never add outside content of "
                    "your own."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.8,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


def generate_offtopic_decline(query):
    """The user's message was classified as off-topic AND a genuine
    factual request (domain_guard.is_factual_offtopic() returned True)
    - e.g. "what's the weather", "who won the Super Bowl". Still
    declines the actual fact - this function must NEVER attempt a real
    answer to the question - but in warmer, more natural phrasing than
    the old flat OFF_TOPIC_MESSAGE constant. The system prompt's
    "CRITICAL RULES" section is the actual safety mechanism (tone is
    free to vary, content is not); OFF_TOPIC_MESSAGE itself is kept as
    the no-API-key fallback below, so a missing key still degrades to
    the same safe, fixed decline as before rather than failing open."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return OFF_TOPIC_MESSAGE

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "The user asked a real question, but it is about "
                    "something entirely unrelated to Wilfrid Laurier "
                    "University (e.g. weather, general trivia, news, "
                    "other schools, coding help, how-to instructions).\n\n"
                    "CRITICAL RULES (never break these, regardless of "
                    "tone):\n"
                    "- Do NOT answer the actual question. Never state any "
                    "real fact, number, date, name, or piece of "
                    "information that would answer what they asked - not "
                    "even a partial, approximate, hedged, or "
                    "'as of my last update' answer.\n"
                    "- Decline warmly and briefly in your own natural "
                    "words - acknowledge you can't help with that "
                    "specific thing, then make your scope clear (e.g. "
                    "'I'm all about WLU', 'I'm here for WLU questions', "
                    "'my focus is WLU' - vary the exact phrasing "
                    "naturally, but always include that idea), then "
                    "invite a WLU question instead. One or two short "
                    ""  # This comma was missing
                    "sentences total.\n"
                    "- Never let a warmer, friendlier tone become an "
                    "excuse to slip in a real answer - the decline must "
                    "stay just as firm as a flat refusal, only phrased "
                    "more naturally."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


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


# Bounded, capability-real follow-up ACTIONS shown as real buttons under
# certain response_types (Google-AI-Mode-style "Would you like to know
# X?", but clicking one directly triggers the deterministic action - no
# free-text reinterpretation involved at all). Deliberately a fixed
# dict, never LLM-generated: an LLM asked to "suggest a natural
# follow-up" has no way to know which follow-up capabilities this app's
# own deterministic routing (resolve_contextual_reference/
# structured_search, retriever.py) can actually resolve, so a free-
# generated suggestion risks promising a capability that doesn't exist
# - exactly the failure mode this dict exists to avoid.
#
# Each action string names a case _resolve_button_action() (below)
# knows how to execute directly against the CAPTURED entity context
# (app.py's _capture_followup_context(), stored per-message as
# "followup_context") - never against "whatever is currently in
# memory", so a button on an older message still acts on the entity
# THAT message was actually about, even if the conversation has since
# moved on to a different course/program/department. Two response
# types map to a LIST of two buttons (course/course_instructors) since
# the old single free-text hint offered an either/or choice a single
# click can't represent - split into two independent, individually
# clickable actions instead:
#   - "prerequisites"/"instructors" build the exact same fixed question
#     template _INTENT_REWRITE_RULES (retriever.py) already rewrites
#     free text into, substituting the captured course code directly,
#     then call structured_search() with it - the same deterministic
#     function the free-text path calls, just without needing to first
#     detect that "prerequisites"/"who teaches" was what was meant.
#   - "coordinator" calls retriever._attempt_coordinator_resolution()
#     directly - already a pure, deterministic memory-lookup with no
#     free-text pattern matching inside it at all.
#   - "first_in_list" calls retriever._attempt_ordinal_resolution()
#     directly, same reasoning.
# A response_type with no verified-safe follow-up (policy, and anything
# not listed) is simply absent - no button is shown, rather than
# guessing at one that might not resolve.
FOLLOWUP_SUGGESTIONS = {
    "course": [
        {"label": "Show prerequisites", "action": "prerequisites"},
        {"label": "Who's teaching it?", "action": "instructors"},
    ],
    "prerequisite": [
        {"label": "Who's teaching it?", "action": "instructors"},
    ],
    "course_instructors": [
        {"label": "Show prerequisites", "action": "prerequisites"},
        {"label": "Tell me about the first one", "action": "first_in_list"},
    ],
    "faculty_profile": [
        {"label": "Who coordinates their department?", "action": "coordinator"},
    ],
    "program": [
        {"label": "Who's coordinating this program?", "action": "coordinator"},
    ],
    "undergraduate_requirements": [
        {"label": "Who coordinates this program?", "action": "coordinator"},
    ],
    "graduate_requirements": [
        {"label": "Who coordinates this program?", "action": "coordinator"},
    ],
    "department_profile": [
        {"label": "Who coordinates this department?", "action": "coordinator"},
    ],
    "faculty_list": [
        {"label": "Tell me about the first one", "action": "first_in_list"},
    ],
    "department_faculty_list": [
        {"label": "Tell me about the first one", "action": "first_in_list"},
    ],
    "research": [
        {"label": "Tell me about the first one", "action": "first_in_list"},
    ],
}


def _is_redundant_full_card_repeat(query, context):
    """True when a "tell me more"/"show me"/etc. follow-up (already in
    FOLLOWUP_PHRASES) resolves to the exact same card content the
    immediately preceding assistant message already showed in full -
    the free-text equivalent of clicking an already-clicked "Show full
    details" button. Confirmed live: structured_search()'s own
    FOLLOWUP MEMORY rewrite has no way to know the entity it's re-
    resolving was already shown in full one turn ago, so it happily
    returns the identical card content again as a brand new, fully
    duplicate message.

    Deliberately scoped to FOLLOWUP_PHRASES specifically, not "any
    query that happens to resolve to the same card" - a user directly
    re-asking "What is CP312?" verbatim is a different, legitimate
    case (maybe they forgot, or want to double check), not the
    "already clicked, clicked again" pattern this guards against.

    Also scoped to _CARD_ON_REQUEST_TYPES specifically - confirmed
    live this matters: response types outside the answer-first
    redesign (e.g. "policy") always have show_card=True and never
    withhold anything to begin with, so "tell me more" re-resolving
    the same policy is the SAME already-tested, legitimate re-confirm
    behavior it's always had - not a redundant repeat of a reveal that
    never happened. An earlier, broader version of this check (any
    show_card=True response type) caught that case too and broke a
    real benchmark question (Policy 12.2 follow-up) that specifically
    exercises this.

    Scans backward for the most recent assistant message with a real
    response_type, not just messages[-2] - confirmed live this matters
    too: a first redundant "more" correctly gets the acknowledgment
    below (response_type=None), but a SECOND consecutive "more" right
    after that would then see messages[-2] as the acknowledgment
    itself (response_type=None, never in _CARD_ON_REQUEST_TYPES) and
    incorrectly conclude nothing was ever shown, re-revealing the same
    card a second time. Skipping past acknowledgment/off-topic/
    conversational messages (response_type=None) to the last message
    that actually resolved to something keeps this a genuine one-way
    reveal no matter how many times "more" is repeated, not just once.
    The search still starts one position back (index -2, not -1) for
    the same reason as before: the current query's own "user" message
    is already appended to st.session_state.messages by the time this
    runs."""

    if normalize_followup_text(query) not in FOLLOWUP_PHRASES:
        return False

    messages = st.session_state.messages

    last = None

    for candidate in reversed(messages[:-1]):

        if candidate["role"] != "assistant":
            continue

        if candidate.get("response_type") is not None:
            last = candidate
            break

    if last is None:
        return False

    return bool(
        last.get("response_type") in _CARD_ON_REQUEST_TYPES
        and last.get("show_card")
        and last.get("content") == context.strip()
    )


# Bare, short affirmatives someone would naturally reply with to the
# assistant's OWN just-asked follow-up question - deliberately NOT the
# same list as FOLLOWUP_PHRASES/_CARD_DETAIL_FOLLOW-up phrasings like
# "yes please"/"show me" (those already mean "show me the full card"
# for a DIFFERENT mechanism, structured_search()'s FOLLOWUP MEMORY
# rewrite, and firing this check for them too would fight that existing
# behavior). Bare "yes" was deliberately left OUT of FOLLOWUP_PHRASES
# itself, earlier in this project, specifically because it's too
# generic a word to safely treat as "show the entity's card" - this is
# a narrower, different case: not "reveal a card", but "elaborate on
# the free-text prose answer you just gave", grounded in that exact
# same answer's own retrieved content, never a fresh search.
_BARE_AFFIRMATIVE_PHRASES = {
    "yes", "yeah", "yea", "yep", "yup", "sure", "ok", "okay",
    "definitely", "absolutely",
}


def _is_bare_affirmative_after_question(query):
    """True when the CURRENT query is a bare affirmative and the
    assistant's own immediately preceding message ended with a genuine
    question mark - i.e. this is very likely a reply to something the
    assistant itself just asked, not an off-topic non-sequitur.
    Confirmed live: "What scholarships are available?" (a real,
    grounded WLU answer that naturally ended "Are you considering
    applying for a specific scholarship...?") followed by "yes" was
    misrouted to the off-topic branch - is_wlu_related("yes") has
    nothing to go on, since the word itself carries no WLU signal at
    all, even though it's clearly answering the assistant's own
    question, not starting a new unrelated one.

    Checks messages[-2], not messages[-1], for the same reason as
    _is_redundant_full_card_repeat() above: the current query's own
    "user" message is already appended to st.session_state.messages by
    the time this runs."""

    if normalize_followup_text(query) not in _BARE_AFFIRMATIVE_PHRASES:
        return False

    messages = st.session_state.messages

    if len(messages) < 2 or messages[-2]["role"] != "assistant":
        return False

    return messages[-2].get("content", "").rstrip().endswith("?")


def _elaborate_on_last_answer(query):
    """Re-grounds a bare "yes"/"sure" reply in the EXACT SAME retrieved
    content that produced the assistant's own preceding question -
    never a fresh search, which could easily surface something
    different for a one-word query with no topical content of its own.
    Returns (answer, context, source, response_type) on success, or
    None when the preceding message has no stored context to elaborate
    on at all (greetings/conversation/off-topic replies never store
    one - see _finalize_response()'s own comment - so there's genuinely
    nothing safe to elaborate on for those, and this correctly declines
    to handle them rather than guessing)."""

    messages = st.session_state.messages

    if len(messages) < 2:
        return None

    last = messages[-2]

    context = last.get("context")

    if not context:
        return None

    response_type = last.get("response_type")

    answer = generate_answer(query, context, response_type)

    return answer, context, last.get("raw_source"), response_type


def _capture_followup_context(memory):
    """Snapshots the entity identifiers a follow-up BUTTON might need,
    immediately after generating an answer - not "whatever's in memory
    whenever the button eventually gets clicked", which could be a
    different course/program/department entirely by then if the
    conversation has since moved on. Captures all three regardless of
    which one the current response_type's buttons actually need (extra
    unused fields are harmless) - simpler than threading response_type-
    specific logic through here, and keeps this function usable
    unchanged if a future response_type needs a different combination.

    coordinator_target/list_id still resolve against CURRENT memory
    when their button is actually clicked (via retriever._attempt_
    coordinator_resolution()/_attempt_ordinal_resolution(), unchanged
    from the free-text path) rather than a snapshotted value - matching
    today's free-text "who coordinates it?"/"tell me about the first
    one" behavior exactly, including its same pre-existing staleness
    edge case (asking about a much older topic after the conversation
    has moved on). course_code is the one identifier captured and used
    directly instead, since "prerequisites"/"instructors" are clearly
    about the one specific course the button's own message was about,
    not whatever course happens to be most recent by the time it's
    clicked.

    Reads entity_id, not _resolve_typed_value()'s display_name -
    confirmed live that a course's display_name is the decorated
    "CP312 - Algorithm Design and Analysis I" form (search_course(),
    retriever.py, records it that way for the pronoun-substitution case
    _resolve_typed_value() is normally used for), not the bare code a
    "What are the prerequisites for {course_code}?" template needs.
    entity_id is the bare code for courses specifically."""

    course_entry = _latest_entity_of_type(memory, "course")

    return {
        "course_code": course_entry["entity_id"] if course_entry else None,
    }


def _resolve_button_action(action, followup_context, memory):
    """Executes a follow-up button's action directly - no free-text
    question is ever constructed from user input and reinterpreted;
    the only "question" text built here is a fixed, known-good template
    with a captured entity identifier substituted in, immediately
    passed to the same deterministic structured_search() the free-text
    path already calls, or a private retriever.py resolver called
    directly. Returns (query_label, context, source, response_type) on
    success, or None if the captured context is missing or the
    resolution attempt fails (e.g. the entity was since removed from
    the corpus) - the caller falls back to a graceful clarification
    message in that case, never a silent no-op."""

    if action == "prerequisites":

        course_code = followup_context.get("course_code")

        if not course_code:
            return None

        question = f"What are the prerequisites for {course_code}?"
        result = structured_search(question, memory)

        if not result:
            return None

        context, source, response_type = result
        return question, context, source, response_type

    if action == "instructors":

        course_code = followup_context.get("course_code")

        if not course_code:
            return None

        question = f"Who has taught {course_code}?"
        result = structured_search(question, memory)

        if not result:
            return None

        context, source, response_type = result
        return question, context, source, response_type

    if action == "coordinator":

        outcome = _attempt_coordinator_resolution(memory)

        if outcome[0] != "resolved":
            return None

        _, context, source, response_type = outcome
        return "Who's coordinating this?", context, source, response_type

    if action == "first_in_list":

        outcome = _attempt_ordinal_resolution(
            "Tell me about the first one.", "first", memory
        )

        if outcome[0] != "resolved":
            return None

        _, context, source, response_type = outcome
        return "Tell me about the first one.", context, source, response_type

    return None


# Answer-first, card-on-request redesign: these four response types are
# the full ENTITY-PROFILE cards (a whole course/person/program/
# department's worth of fields at once) - the ones dense enough that
# leading with the short grounded summary, then only expanding to the
# full card on request, is worth the extra turn. Deliberately narrower
# than _COURSE_RESPONSE_TYPES/_PROGRAM_RESPONSE_TYPES (renderer.py):
# "prerequisite"/"course_instructors"/"coordinator"/"undergraduate_
# requirements"/"faculty_list"/... are already narrow, single-fact or
# list-shaped answers, not a dense multi-field profile, so they keep
# rendering immediately exactly as before.
_CARD_ON_REQUEST_TYPES = {
    "course",
    "faculty_profile",
    "program",
    "department_profile",
}

# Exception to the above: a question that was already broad/open-ended
# shows the full card immediately - asking "want the full details?"
# right after someone explicitly asked for everything would be an
# annoying extra round trip, not a helpful pause. Deliberately matched
# against the phrase alone, not tied to any specific response_type,
# since the same phrasing works regardless of which of the four types
# ends up answering it.
_BROAD_DETAIL_REQUEST_PATTERN = re.compile(
    r"\beverything\b|\ball\s+(?:the\s+)?details\b|\bfull\s+details\b|"
    r"\bcomplete\s+details\b|\bfull\s+information\b|\bin\s+detail\b|"
    r"\bin-depth\b|\bcomprehensive\b|\ball\s+about\b|\bfull\s+profile\b",
    re.IGNORECASE
)

# The natural ways someone responds to "Want the full details?" (e.g.
# "show me", "yes please") are added directly to FOLLOWUP_PHRASES
# itself (retriever.py), not a separate list here - structured_search()'s
# own FOLLOWUP MEMORY block re-resolves any FOLLOWUP_PHRASES member
# against the most recently established entity and re-runs the full
# structured lookup for it, which is exactly "fetch this entity's full
# details again". A bare "show me" names no entity of its own, so
# without that shared rewrite it would never resolve to anything -
# checking a locally-defined phrase set here, without also teaching
# retriever.py's rewrite the same phrases, would make this condition
# true while structured_search() had already failed to produce a
# course/faculty_profile/program/department_profile result at all.
# Still available as a fallback for anyone who types "tell me more"
# unprompted, with no button context at all - only removed as the
# PRIMARY path once a button already covers the same action.


def _finalize_response(query, answer, source, response_type, memory, context=None):
    """Everything from citation enrichment through message storage,
    once answer/source/response_type are already known - shared by both
    the free-text turn (structured_search()/resolve_contextual_
    reference()/hybrid_search(), below) and button-driven follow-up
    actions (_resolve_button_action() above), which differ only in HOW
    they arrived at those three values, never in what happens once they
    have them. Does not render anything itself - callers render inside
    their own st.chat_message(...) block, since the live free-text path
    also needs to clear its loading placeholder first, which a button
    click never has. Returns the message dict that was appended to
    st.session_state.messages, so a caller that needs it (none do
    today) could inspect it further.

    context, when the caller has it (the raw retrieved text that
    grounded `answer`, not the enriched citation `source` becomes below)
    is stored on the message as-is, alongside the RAW `source` string
    (before citation.build_citation() below turns it into the enriched
    {"date", "sources"} shape) - both purely for
    _elaborate_on_last_answer()'s benefit (a bare "yes"/"sure" after
    this answer ends in a question), which needs to re-ground a follow-
    up in the exact same retrieved content rather than either inventing
    new information or running a fresh, possibly-different search.
    Callers that don't have context (greetings, conversation, off-topic
    replies) simply leave it None - there's nothing to elaborate on for
    those anyway."""

    raw_source = source

    source = citation.build_citation(source, response_type, answer)

    summary = generate_grounded_summary(
        query,
        answer,
        response_type
    )

    show_card = True

    if response_type in _CARD_ON_REQUEST_TYPES and summary:

        normalized_query = normalize_followup_text(query)

        is_broad_request = bool(
            _BROAD_DETAIL_REQUEST_PATTERN.search(query)
        )
        is_detail_followup = normalized_query in FOLLOWUP_PHRASES

        show_card = is_broad_request or is_detail_followup

    # followup is always either None or a list of {"label", "action"}
    # button specs now - "reveal_card" is a special action
    # _render_followup_buttons() below handles in place (flips this
    # same message's show_card rather than creating a new Q&A pair),
    # never routed through _resolve_button_action().
    if not show_card:
        followup = [{"label": "Show full details", "action": "reveal_card"}]
    else:
        followup = (
    # End of the previous use of followup
            FOLLOWUP_SUGGESTIONS.get(response_type) if response_type else None
        )

    followup_context = _capture_followup_context(memory) if followup else None

    message = {
        "role": "assistant",
        "content": answer,
        "source": source,
        "response_type": response_type,
        "summary": summary,
        "followup": followup,
        "followup_context": followup_context,
    }  # Closing brace for message object
        "context": context,
        "raw_source": raw_source
    } # Added closing brace
    # Using message_index from context to track expansion state
    message_index = len(st.session_state.messages)
    st.session_state[message_index] = st.session_state.get(message_index, False)

    if not st.session_state[message_index]:
        if st.button("Show Details", key=f"show-{message_index}"):
            st.session_state[message_index] = True
    else:
        if st.button("Hide Details", key=f"hide-{message_index}"):
            st.session_state[message_index] = False
    
    # Code rendering details here based on the expansion state
    }

    st.session_state.messages.append(message)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return message


def _render_followup_buttons(message_index, message):
    """Renders each follow-up action (or the answer-first redesign's
    "show full details" prompt) as a real button - clicking one
    directly triggers its mapped deterministic action and updates
    conversation state in place, with no free-text reinterpretation
    involved anywhere in this path. Unique key per (message, action)
    pair so multiple messages' buttons - and the two buttons "course"/
    "course_instructors" map to - never collide; message_index is
    stable across reruns since messages are only ever appended, never
    reordered or removed."""

    followup = message.get("followup")

    if not followup:
        return

    for action_index, suggestion in enumerate(followup):

        button_key = f"followup-{message_index}-{action_index}"

        if not st.button(suggestion["label"], key=button_key):
            continue

        if suggestion["action"] == "reveal_card":

            message["show_card"] = True

            # Confirmed live: without this, the "Show full details"
            # button stayed in `followup` unchanged after being
            # clicked, so it kept reappearing (now next to an already-
            # fully-shown card) and clicking it again was a no-op
            # re-render, not a one-way reveal. Replaced with whatever
            # FOLLOWUP_SUGGESTIONS actually maps to this response_type
            # once the card IS shown - the exact same computation
            # _finalize_response() does when show_card is True from the
            # start - or None if there isn't one, rather than leaving a
            # stale action behind.
            message["followup"] = (
                FOLLOWUP_SUGGESTIONS.get(message.get("response_type"))
                if message.get("response_type") else None
            )

            st.rerun()
            return

        followup_context = message.get("followup_context") or {}

        resolved = _resolve_button_action(
            suggestion["action"], followup_context, st.session_state.memory
        )

        if not resolved:

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, I couldn't look that up right now. "
                        "Feel free to ask directly instead."
                    ),
                    "source": None,
                    "response_type": None,
                    "summary": None,
                    "followup": None,
                    "followup_context": None,
                    "show_card": True
                }
            )
            st.rerun()
            return

        query_label, context, source, response_type = resolved

        answer = generate_answer(query_label, context, response_type)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query_label
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query_label
            }
        )

        _finalize_response(
            query_label, answer, source, response_type,
            st.session_state.memory, context
        )

        st.rerun()
        return


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
    /* ---- Color system --------------------------------------------
       Dark purple sidebar gradient (unchanged), neutral near-white
       main canvas (--wlu-paper) - not a purple wash - so the app reads
       as a chat tool, not a marketing page. --wlu-purple-light is kept
       only for smaller accents (e.g. the user-message tint mixed with
       --wlu-purple-soft) - it is no longer a large fill area. Every
       color used anywhere in this file (and in renderer.py's card
       CSS, via the same variable names) draws from this single set -
       no other hex value should appear as a fill for text,
       backgrounds, or borders outside this block.
       Contrast verified (WCAG relative-luminance formula): --wlu-ink
       on --wlu-paper is 16.9:1, --wlu-ink-muted on white/paper is
       5.9:1 - the vast majority of the app's text now uses this simple
       pairing instead of juggling contrast against a colored canvas.
       White text still passes on --wlu-purple (10.6:1) and
       --wlu-purple-dark (13.4:1) for the hero/sidebar. --wlu-gold
       passes against those two darker purples (8.5:1) as text, and
       everywhere as a non-text UI accent (focus rings, thin borders,
       >=3:1) - but gold text/icons must never sit on white/paper
       (1.5:1, fails badly) or on --wlu-purple-light (3.5:1, fails for
       text). */
    --wlu-purple-dark: #3E1C66;
    --wlu-purple: #522687;
    --wlu-purple-light: #8454A0;
    --wlu-gold: #FCC707;
    --wlu-ink: #1A1526;
    --wlu-ink-muted: #6B5F7A;
    --wlu-paper: #FAF8FC;
    --wlu-card-bg: #FFFFFF;
    --wlu-purple-soft: #EFEBF4;
    --wlu-border: #E4DCEF;
    --wlu-online: #1F9D55;
    /* --wlu-purple-text: a separate token from --wlu-purple, used
       wherever purple is the TEXT color on a paper/card surface
       (section titles, "Try asking", card-footer links) rather than a
       background fill. In light mode this is identical to --wlu-purple
       (10.6:1 on white - fine as-is); dark mode overrides it to a
       brighter purple, since --wlu-purple itself only reaches 1.7:1
       against a near-black surface as text - recomputed, not assumed,
       see the dark-mode block below. */
    --wlu-purple-text: var(--wlu-purple);

    /* ---- Typography system -----------------------------------------
       One header font (Poppins) + one body font (Inter), four
       deliberate size tiers used everywhere - hero / section-header /
       body / caption. Nothing outside this scale. */
    --wlu-font-head: 'Poppins', 'Inter', sans-serif;
    --wlu-font-body: 'Inter', -apple-system, "Segoe UI", sans-serif;
    --wlu-fs-hero: 2.25rem;      /* 36px / 700 / hero title only */
    --wlu-fs-h2: 1.375rem;       /* 22px / 700 / every section header */
    --wlu-fs-body: 1rem;         /* 16px / 400-500 / primary reading copy */
    --wlu-fs-body-sm: 0.9375rem; /* 15px / 400-500 / secondary inline copy */
    --wlu-fs-caption: 0.8125rem; /* 13px / 500 / meta, labels, footers */
    --wlu-fs-micro: 0.75rem;     /* 12px / 700 / uppercase eyebrow tags */

    /* ---- Spacing system: strict 8px scale --------------------------- */
    --wlu-sp-1: 0.5rem;   /* 8px */
    --wlu-sp-2: 1rem;     /* 16px */
    --wlu-sp-3: 1.5rem;   /* 24px */
    --wlu-sp-4: 2rem;     /* 32px */
    --wlu-sp-6: 3rem;     /* 48px */

    /* ---- Shared elevation, radius, and motion tokens ---------------
       One radius, one shadow pair (rest/hover), one transition timing
       - applied identically to every card/panel/input in the app.
       Deliberate exceptions: suggested-question chips stay pill-shaped
       (a distinct control type, not a card) and circular badges/dots
       stay circular (icon/indicator shapes, not cards). */
    --wlu-radius: 16px;
    --wlu-shadow-rest: 0 4px 14px rgba(26, 21, 38, 0.12);
    --wlu-shadow-hover: 0 10px 28px rgba(26, 21, 38, 0.18);
    --wlu-transition: 180ms ease;
    /* A thin (2px) branded ring, not a thick amber one that reads as a
       validation error. Solid --wlu-purple carries the actual contrast
       (10.6:1 on white/paper - passes on its own); the gold layer
       outside it is a low-opacity flourish only, never load-bearing
       for visibility, since gold alone fails contrast against white/
       paper (1.5:1). */
    --wlu-focus-ring: 0 0 0 2px var(--wlu-purple), 0 0 0 4px rgba(252, 199, 7, 0.25);
}

html, body, [class*="css"] {
    font-family: var(--wlu-font-body);
    font-size: var(--wlu-fs-body);
    line-height: 1.55;
    color: var(--wlu-ink);
}

/* Custom list styling, injected once here rather than per-card in
   renderer.py's _CARD_CSS, so it applies globally to every markdown-
   rendered list in the app - structured-card body text and generic/
   vector answers alike - as one consistent system, not two. Overrides
   the default black serif numbers / hollow-circle browser bullets
   with on-brand markers via ::marker (color/content, not a background-
   image or extra markup - no DOM structure change, so this never
   touches what _extract_labeled_field() parses). */
ul, ol {
    padding-left: 1.3em;
    margin: var(--wlu-sp-1) 0;
}
ul li, ol li {
    margin-bottom: var(--wlu-sp-1);
    line-height: 1.6;
}
ul li:last-child, ol li:last-child {
    margin-bottom: 0;
}
ul > li::marker {
    content: "●  ";
    color: var(--wlu-purple-text);
    font-size: 0.7em;
}
ol > li::marker {
    color: var(--wlu-purple-text);
    font-weight: 700;
}
/* Sub-lists (nested one level in) - indented further, a lighter
   secondary marker and muted color instead of the browser default
   hollow-circle, so nesting reads as "less important," not just
   "further right." */
ul ul, ol ul, ul ol, ol ol {
    padding-left: 1.2em;
    margin: 0.35em 0;
}
ul ul > li::marker {
    content: "–  ";
    color: var(--wlu-ink-muted);
    font-size: 0.85em;
}
ol ul > li::marker {
    content: "–  ";
    color: var(--wlu-ink-muted);
    font-size: 0.85em;
}

@keyframes wluFadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes wluPulse {
    0%, 80%, 100% { opacity: 0.25; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}

/* Typing indicator - three pulsing dots shown in the assistant's own
   avatar/bubble styling (not a bare Streamlit spinner), so a response
   that's still generating still looks like part of the same card
   system rather than a generic loading state. */
.wlu-typing {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.3rem 0;
}
.wlu-typing span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--wlu-purple);
    animation: wluPulse 1.2s ease-in-out infinite;
}
.wlu-typing span:nth-child(2) { animation-delay: 0.15s; }
.wlu-typing span:nth-child(3) { animation-delay: 0.3s; }

/* Main canvas: flat, neutral near-white (--wlu-paper) - a chat surface,
   not a colored marketing panel. The prior large ambiguous watermark
   shape (it read as an unclear triangle/blob rather than a
   recognizable leaf) is removed rather than patched; the one leaf
   accent in the app now lives small and unambiguous in the sidebar
   brand mark instead. */
.stApp {
    background-color: var(--wlu-paper);
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
    background: linear-gradient(90deg, var(--wlu-purple-dark) 0%, var(--wlu-gold) 50%, var(--wlu-purple) 100%);
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
    background: rgba(62, 28, 102, 0.35);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--wlu-purple-dark);
}

.block-container {
    padding-top: var(--wlu-sp-4);
    max-width: 760px;
}

/* The chat input lives in a separate fixed-position container
   (Streamlit's own "bottom block"), not inside .block-container, so it
   has its own default max-width/centering that doesn't automatically
   match the column above it - overridden here so the input stays
   visually aligned with the hero/cards/messages rather than sitting
   wider or off-center from them. */
div[data-testid="stBottomBlockContainer"] {
    max-width: 760px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

h1, h2, h3, h4 {
    font-family: var(--wlu-font-head);
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: -0.01em;
}
.block-container h4 {
    color: var(--wlu-purple-text);
    font-size: var(--wlu-fs-h2);
    margin: var(--wlu-sp-1) 0 var(--wlu-sp-2);
}
.block-container hr {
    border-color: var(--wlu-border);
    margin: var(--wlu-sp-3) 0;
}

/* Hero */
.wlu-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--wlu-purple) 0%, var(--wlu-purple-dark) 100%);
    border-radius: var(--wlu-radius);
    padding: var(--wlu-sp-6);
    margin-bottom: var(--wlu-sp-3);
    box-shadow: var(--wlu-shadow-hover);
    animation: wluFadeUp 0.5s ease both;
}
.wlu-hero::before {
    content: '';
    position: absolute;
    top: -70px;
    right: -70px;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(252, 199, 7, 0.28) 0%, rgba(252, 199, 7, 0) 70%);
    pointer-events: none;
}
.wlu-hero-badge {
    position: absolute;
    top: var(--wlu-sp-4);
    right: var(--wlu-sp-4);
    width: 48px;
    height: 48px;
    border-radius: 50%;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.08);
    border: 2px solid var(--wlu-gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    line-height: 1;
    box-shadow: 0 0 0 4px rgba(252, 199, 7, 0.14), 0 4px 14px rgba(0, 0, 0, 0.28);
}
.wlu-hero h1 {
    position: relative;
    font-size: var(--wlu-fs-hero);
    font-weight: 700;
    line-height: 1.2;
    color: #FFFFFF;
    margin: 0 5.5rem var(--wlu-sp-1) 0;
    text-shadow: 0 2px 14px rgba(0, 0, 0, 0.25);
}
.wlu-hero .wlu-tagline {
    position: relative;
    font-size: var(--wlu-fs-body);
    font-weight: 600;
    color: var(--wlu-gold);
    margin: 0 0 var(--wlu-sp-2);
}
.wlu-hero .wlu-desc {
    position: relative;
    font-size: var(--wlu-fs-body-sm);
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.9);
    max-width: 62ch;
    margin: 0;
}

/* Welcome card */
.wlu-welcome {
    background: var(--wlu-card-bg);
    border: 1px solid var(--wlu-border);
    border-left: 4px solid var(--wlu-gold);
    border-radius: var(--wlu-radius);
    padding: var(--wlu-sp-3);
    /* Small on its own (--wlu-sp-1) rather than a full section gap
       (--wlu-sp-3) - the divider immediately following this card
       already carries its own --wlu-sp-3 top margin
       (.block-container hr below), and the two were compounding into
       a gap that read as accidental dead space rather than one
       considered --wlu-sp-4 (32px) section break. */
    margin-bottom: var(--wlu-sp-1);
    box-shadow: var(--wlu-shadow-rest);
    animation: wluFadeUp 0.55s ease 0.08s both;
}
.wlu-welcome .wlu-welcome-title {
    font-family: var(--wlu-font-head);
    font-weight: 700;
    color: var(--wlu-purple-text);
    font-size: var(--wlu-fs-h2);
}
.wlu-welcome p {
    margin: var(--wlu-sp-1) 0 0;
    color: var(--wlu-ink-muted);
    font-size: var(--wlu-fs-body);
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
    background-image: linear-gradient(165deg, var(--wlu-purple-dark) 0%, var(--wlu-purple) 100%);
}
section[data-testid="stSidebar"] h3 {
    font-family: var(--wlu-font-head) !important;
    color: #FFFFFF !important;
    font-size: var(--wlu-fs-h2) !important;
    font-weight: 700 !important;
    margin-top: var(--wlu-sp-1) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown {
    font-size: var(--wlu-fs-body);
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.85);
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.18);
    margin: var(--wlu-sp-3) 0;
}
section[data-testid="stSidebar"] ul {
    list-style: none;
    padding-left: 0;
    margin: var(--wlu-sp-1) 0 0;
}
section[data-testid="stSidebar"] li {
    position: relative;
    padding-left: var(--wlu-sp-3);
    margin-bottom: var(--wlu-sp-1);
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

/* New Conversation button - solid gold so it reads as the sidebar's one
   primary action, clearly distinct from the white suggested-question
   chips in the main canvas. Purple-dark text/icon on gold: 8.5:1,
   passes easily (the pairing already verified safe in the color-system
   comment above). */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: var(--wlu-gold) !important;
    color: var(--wlu-purple-dark) !important;
    border: none !important;
    font-weight: 700 !important;
    margin-bottom: var(--wlu-sp-2) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #E6B406 !important;
    color: var(--wlu-purple-dark) !important;
    border: none !important;
    transform: translateY(-1px);
}

/* Brand lockup - a horizontal wordmark + leaf-mark treatment in the
   spirit of WLU's own "LAURIER [leaf]" logo lockup, but original
   typography and an original leaf shape, not a reproduction of their
   registered logo asset. */
.wlu-brand {
    text-align: left;
    padding: var(--wlu-sp-1) 0 var(--wlu-sp-3);
    animation: wluFadeUp 0.45s ease both;
}
.wlu-brand-row {
    display: flex;
    align-items: center;
    gap: var(--wlu-sp-1);
}
.wlu-brand-word {
    font-family: var(--wlu-font-head);
    font-weight: 700;
    font-size: 1.65rem;
    letter-spacing: 0.01em;
    color: #FFFFFF;
    line-height: 1;
}
.wlu-brand-mark {
    font-size: 1.6rem;
    line-height: 1;
    flex-shrink: 0;
    filter: drop-shadow(0 0 6px rgba(252, 199, 7, 0.35));
}
.wlu-brand-tagline {
    font-family: var(--wlu-font-body);
    font-weight: 600;
    font-size: var(--wlu-fs-micro);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.75);
    margin-top: var(--wlu-sp-1);
}
.wlu-status-card {
    position: relative;
    overflow: hidden;
    background: var(--wlu-card-bg);
    border: 1px solid var(--wlu-border);
    border-radius: var(--wlu-radius);
    padding: var(--wlu-sp-2) var(--wlu-sp-2) var(--wlu-sp-1);
    margin-bottom: var(--wlu-sp-2);
    box-shadow: var(--wlu-shadow-hover);
    animation: wluFadeUp 0.5s ease 0.05s both;
}
.wlu-status-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--wlu-purple-dark), var(--wlu-purple));
}
.wlu-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0;
    font-size: var(--wlu-fs-caption);
}
.wlu-status-row + .wlu-status-row {
    border-top: 1px solid var(--wlu-border);
}
.wlu-status-row .wlu-label {
    color: var(--wlu-ink-muted);
    display: flex;
    align-items: center;
    font-weight: 500;
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
    margin-right: var(--wlu-sp-1);
    box-shadow: 0 0 0 3px rgba(31, 157, 85, 0.15);
}
.wlu-version {
    text-align: center;
    font-size: var(--wlu-fs-micro);
    color: rgba(255, 255, 255, 0.6);
    margin-top: var(--wlu-sp-3);
    padding-top: var(--wlu-sp-2);
    border-top: 1px solid rgba(255, 255, 255, 0.18);
}

/* Chat bubbles. Streamlit's chat message markup does NOT expose a
   role-specific data-testid on the message or its avatar (confirmed by
   inspecting the live DOM - stChatMessageAvatarUser/Assistant don't
   exist in this Streamlit version, so the two rules that used to
   target them were dead selectors that never matched anything). The
   one reliable, stable role signal is the inner content's own
   aria-label ("Chat message from user"/"...from assistant"), used
   here via :has() to target the outer bubble.

   Each bubble is a compact, role-aligned block rather than a full-
   width row: user messages are right-aligned and capped at 70% of the
   column (compact, since questions are short); assistant messages are
   left-aligned and allowed more width (answers are longer). */
div[data-testid="stChatMessage"] {
    background: var(--wlu-paper);
    border: 1px solid var(--wlu-border);
    border-radius: var(--wlu-radius);
    box-shadow: var(--wlu-shadow-rest);
    transition: box-shadow var(--wlu-transition);
    animation: wluFadeUp var(--wlu-transition) both;
    width: fit-content;
    max-width: 88%;
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    background: var(--wlu-purple-soft);
    flex-direction: row-reverse;
    margin-left: auto;
    margin-right: 0;
    max-width: 70%;
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
    margin-right: auto;
    margin-left: 0;
}
div[data-testid="stChatMessage"]:hover {
    box-shadow: var(--wlu-shadow-hover);
}

/* Avatars - a small circular badge with a custom geometric glyph
   (person silhouette / graduation cap), not stock emoji. Streamlit
   still receives an emoji via avatar= (a value it requires), but its
   native glyph is hidden here (font-size: 0) and replaced with the
   glyph below via ::before - the avatar div's own box is a fixed
   32x32px flex-centered square regardless of its text content, so
   hiding the glyph text doesn't collapse it. */
div[data-testid="stChatMessage"] > div:first-child {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-size: 0;
    position: relative;
    box-shadow: 0 2px 6px rgba(26, 21, 38, 0.18);
    /* The message row top-aligns the avatar with the text block - a
       fixed-height avatar starting at the same y as a shorter first
       text line makes the avatar's own center sit measurably lower
       than the first line's center (confirmed live: 5.6px at this
       avatar's old 36px size, against the body text's 24.8px line-
       height - 16px font-size * the global 1.55 line-height). Nudging
       the avatar up by half the height difference between it and one
       text line re-centers it against that first line specifically,
       not the whole (possibly multi-line) message block. A fixed
       px value, not em/rem, deliberately: this rule's own
       font-size: 0 above would make 1em resolve to 0 here, not the
       body's actual size. */
    margin-top: calc((24.8px - 40px) / 2);
}
div[data-testid="stChatMessage"] > div:first-child::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background-repeat: no-repeat;
    background-position: center;
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {
    background: linear-gradient(135deg, var(--wlu-purple-light), var(--wlu-purple));
    /* The shared -7.6px margin-top above (calibrated against the
       assistant bubble, which it now centers exactly, confirmed live)
       left the user avatar 3.79px too high - the row-reverse layout
       used for user bubbles measurably affects the text block's own
       vertical offset versus the assistant's normal row direction, so
       the two need slightly different corrections to both land on
       their own first line's center. */
    margin-top: -1.9px;
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child::before {
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%3E%3Ccircle%20cx='12'%20cy='8'%20r='4'%20fill='white'/%3E%3Cpath%20d='M4,20%20A8,8%200%200,1%2020,20%20Z'%20fill='white'/%3E%3C/svg%3E");
    background-size: 20px 20px;
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) > div:first-child {
    background: linear-gradient(135deg, var(--wlu-gold), #D68F00);
}
div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) > div:first-child::before {
    background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%3E%3Cpath%20d='M12,4%20L22,9%20L12,14%20L2,9%20Z'%20fill='%233E1C66'/%3E%3Crect%20x='8'%20y='10'%20width='8'%20height='5'%20rx='1'%20fill='%233E1C66'%20opacity='0.85'/%3E%3Cline%20x1='19'%20y1='9'%20x2='19'%20y2='16'%20stroke='%233E1C66'%20stroke-width='1.5'/%3E%3Ccircle%20cx='19'%20cy='16.5'%20r='1.3'%20fill='%233E1C66'/%3E%3C/svg%3E");
    background-size: 22px 22px;
}

/* Chat input */
div[data-testid="stChatInput"] {
    background: var(--wlu-paper) !important;
    border-radius: var(--wlu-radius) !important;
    border: 1px solid var(--wlu-border) !important;
    box-shadow: var(--wlu-shadow-rest);
    transition: box-shadow var(--wlu-transition), border-color var(--wlu-transition);
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--wlu-purple) !important;
    box-shadow: var(--wlu-focus-ring);
}
/* The wrapper's box-shadow above is the intended ring. Streamlit's own
   BaseWeb textarea internals apply their own default red focus/
   validation border several levels deep inside this wrapper -
   unrelated to this app's palette - which would otherwise show through
   as a second, off-brand ring alongside the intended one. */
div[data-testid="stChatInput"] div {
    border-color: transparent !important;
}
button[data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background: var(--wlu-purple) !important;
    color: #FFFFFF !important;
    transition: background var(--wlu-transition);
}
button[data-testid="stChatInputSubmitButton"]:not(:disabled):hover {
    background: var(--wlu-purple-dark) !important;
}
button[data-testid="stChatInputSubmitButton"]:focus-visible {
    outline: none;
    box-shadow: var(--wlu-focus-ring);
}

/* Suggested-question buttons */
.stButton > button {
    border-radius: var(--wlu-radius) !important;
    border: 1px solid var(--wlu-border) !important;
    background: var(--wlu-card-bg) !important;
    font-family: var(--wlu-font-body) !important;
    font-size: var(--wlu-fs-body-sm) !important;
    font-weight: 500 !important;
    color: var(--wlu-ink) !important;
    padding: 0.65rem var(--wlu-sp-2) !important;
    box-shadow: var(--wlu-shadow-rest) !important;
    transition: transform var(--wlu-transition), box-shadow var(--wlu-transition), border-color var(--wlu-transition), color var(--wlu-transition), background var(--wlu-transition) !important;
    animation: wluFadeUp 0.4s ease both;
}
.stButton > button:hover {
    border-color: var(--wlu-purple) !important;
    color: var(--wlu-purple-text) !important;
    background: var(--wlu-paper) !important;
    transform: translateY(-2px);
    box-shadow: var(--wlu-shadow-hover) !important;
}
.stButton > button:active {
    transform: translateY(0);
}
.stButton > button:focus-visible {
    outline: none;
    box-shadow: var(--wlu-focus-ring) !important;
}

/* Universal focus-visible fallback: any other native interactive
   element (links, etc.) gets the same ring, same timing, rather than
   the browser default outline on some elements and nothing on others. */
a:focus-visible {
    outline: none;
    box-shadow: var(--wlu-focus-ring);
    border-radius: 4px;
}

/* Follow-up suggestion caption (app.py's FOLLOWUP_SUGGESTIONS, rendered
   via st.caption() immediately after a response card) - visually
   attached as a footer row of the card above it, rather than a stray
   floating line, by styling the caption's own container to continue
   the card's border/background and pulling it flush against the
   card's bottom edge. Targets the *next* stElementContainer sibling of
   whichever one holds a .wlu-card - a structural relationship, not a
   change to renderer.py's own markup. */
div[data-testid="stElementContainer"]:has(.wlu-card)
  + div[data-testid="stElementContainer"]:has([data-testid="stCaptionContainer"]) {
    margin-top: calc(-1 * var(--wlu-sp-1));
}
div[data-testid="stElementContainer"]:has(.wlu-card)
  + div[data-testid="stElementContainer"] [data-testid="stCaptionContainer"] {
    background: var(--wlu-paper);
    border: 1px solid var(--wlu-border);
    border-top: none;
    border-radius: 0 0 var(--wlu-radius) var(--wlu-radius);
    padding: var(--wlu-sp-1) var(--wlu-sp-3);
    box-shadow: var(--wlu-shadow-rest);
}
div[data-testid="stElementContainer"]:has(.wlu-card)
  + div[data-testid="stElementContainer"] [data-testid="stCaptionContainer"] p {
    margin: 0;
    color: var(--wlu-ink-muted);
    font-size: var(--wlu-fs-caption);
}
</style>
"""


# Dark-mode token overrides, hooked into Streamlit's own native theme
# toggle (the hamburger menu, top right) rather than building a
# separate one.
#
# st.get_option("theme.base") was tried first and confirmed NOT to work
# for this: toggling the menu's Light/Dark choice doesn't trigger a
# script rerun at all (Streamlit repaints its own built-in widgets
# purely client-side by rewriting its own stylesheet rules in place),
# and even forcing a rerun afterward (verified live, via clicking
# another button) still returned the original startup value - this
# option only reflects the config-level theme, never a live in-app
# override, in this Streamlit version.
#
# What DOES change live is genuinely observable: document.body's own
# computed background-color, which Streamlit rewrites in place -
# confirmed live at rgb(14, 17, 23) on dark, rgb(255, 255, 255)-ish on
# light. THEME_DETECTOR_JS (below) reads that color's luminance from
# inside a components.v1.html() iframe (same-origin as the main app,
# so window.parent.document is reachable) and sets a data-theme
# attribute on the top-level <html> element accordingly, polling
# briefly since Streamlit's rewrite isn't exposed as an observable
# attribute/class change. This is the "data-theme approach" this task
# explicitly allows as the fallback when st.get_option doesn't pan
# out - CSS below keys off :root[data-theme="dark"], which targets the
# same <html> element the script sets the attribute on.
#
# Only tokens that actually need different values for a dark surface
# are overridden - --wlu-purple/--wlu-purple-dark/--wlu-purple-light/
# --wlu-gold are unchanged, since the hero and sidebar already use them
# as background fills with white text (unaffected by overall theme),
# and --wlu-gold already passes as text against a near-black surface
# (11.6:1) with no brightening needed.
#
# Contrast recomputed for these dark pairings specifically (WCAG
# relative-luminance formula), not assumed from the light-mode numbers:
#   --wlu-ink (F1EEF5) on --wlu-paper (17131F): 15.9:1
#   --wlu-ink-muted (B6ACC4) on --wlu-paper: 8.4:1
#   --wlu-ink on --wlu-card-bg (211B2E): 14.9:1
#   --wlu-gold on --wlu-paper: 11.6:1
#   --wlu-purple-text (B08FD4, replaces --wlu-purple wherever purple is
#     used as TEXT rather than a background - the unchanged --wlu-purple
#     itself is only 1.7:1 as text on a dark surface, well under
#     threshold) on --wlu-paper: 6.7:1
#   --wlu-purple-text on --wlu-purple-soft (the dark-mode user-bubble
#     tint, 2D2440): 5.4:1
DARK_MODE_CSS = """
<style>
:root[data-theme="dark"] {
    --wlu-paper: #17131F;
    --wlu-card-bg: #211B2E;
    --wlu-ink: #F1EEF5;
    --wlu-ink-muted: #B6ACC4;
    --wlu-border: #3D3450;
    --wlu-purple-soft: #2D2440;
    --wlu-purple-text: #B08FD4;
}
</style>
"""

THEME_DETECTOR_JS = """
<script>
(function() {
    function applyTheme() {
        try {
            var doc = window.parent.document;
            var bg = getComputedStyle(doc.body).backgroundColor;
            var m = bg.match(/[\\d.]+/g);
            if (!m) return;
            var luminance = 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
            var isDark = luminance < 128;
            var current = doc.documentElement.getAttribute('data-theme');
            var target = isDark ? 'dark' : 'light';
            if (current !== target) {
                doc.documentElement.setAttribute('data-theme', target);
            }
        } catch (e) {}
    }
    applyTheme();
    // Streamlit rewrites its own stylesheet rule in place on a theme
    // toggle (not an attribute/class change on body), so there's
    // nothing to attach a MutationObserver to directly - a short poll
    // is the reliable option, and cheap enough for a single boolean
    // color check every 600ms.
    setInterval(applyTheme, 600);
})();
</script>
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

# Always injected (unlike a plain :root block, this one only takes
# effect once data-theme="dark" is actually set below) - see the
# comment above DARK_MODE_CSS for why this couldn't be a simple
# st.get_option() conditional.
st.markdown(
    DARK_MODE_CSS,
    unsafe_allow_html=True
)

components.html(THEME_DETECTOR_JS, height=0)

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
                <span class="wlu-brand-mark">🍁</span>
            </div>
            <div class="wlu-brand-tagline">Hybrid RAG Assistant</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "＋ New Conversation",
        key="new_conversation_button",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.memory = create_memory()
        st.session_state.pending_query = None
        st.rerun()

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

for _message_index, msg in enumerate(st.session_state.messages):

    with st.chat_message(
        msg["role"],
        avatar=USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    ):

        # A user message has no response_type/source/summary/followup -
        # it's never been through render_response() at all, live: the
        # user-input block below renders it with a plain st.markdown(),
        # no card or eyebrow label. This loop re-runs on every script
        # rerun (every new interaction), so without this branch, once a
        # user message stops being the newest turn it would be replayed
        # straight through render_response() instead - which, having no
        # response_type to dispatch on, falls through to render_generic()
        # and wrongly prepends the "Answer" eyebrow label to the user's
        # own question. Kept in exact sync with the live rendering path
        # below on purpose, not merely similar to it.
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_response(
                msg.get("response_type"),
                msg["content"],
                msg.get("source"),
                msg.get("summary"),
                # Defaults True for messages stored before this field
                # existed - a card that was already shown once, in an
                # older session, must keep being shown on replay, never
                # retroactively hidden.
                msg.get("show_card", True),
            )

            _render_followup_buttons(_message_index, msg)


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
        "user",
        avatar=USER_AVATAR
    ):
        st.markdown(
            query
        )

    # A card-styled typing indicator (three pulsing dots, not a bare
    # Streamlit spinner) shown in its own placeholder while retrieval/
    # generation runs below - st.empty() lets it be cleared in place
    # once the real response is ready, in both the success and error
    # paths, rather than lingering or stacking with the real message.
    loading_placeholder = st.empty()

    with loading_placeholder.container():
        with st.chat_message(
            "assistant",
            avatar=ASSISTANT_AVATAR
        ):
            st.markdown(
                '<div class="wlu-typing"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True
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

            context = None
            source = None
            response_type = None

        # normal conversation
        elif is_conversation(query):

            answer = generate_chat_response(
                query
            )

            context = None
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

            if _is_redundant_full_card_repeat(query, context):

                answer = (
                    "You're already seeing the full details for that - "
                    "let me know if there's something else you'd like "
                    "to know!"
                )
                source = None
                response_type = None

            else:

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
                context = None
                source = None
                response_type = None

        # A bare, short affirmative ("yes"/"sure"/"ok") replying to the
        # assistant's OWN just-asked follow-up question - checked here,
        # before the off-topic branch below, specifically because
        # that's the exact branch this used to misfire into. Confirmed
        # live: "What scholarships are available?" (ending "Are you
        # considering applying for a specific scholarship...?")
        # followed by "yes" had nothing WLU-specific in "yes" itself
        # for is_wlu_related() to recognize, so it fell through to
        # off-topic and got answered as if it were unrelated small talk
        # ("it sounds like you're feeling something!"). See
        # _elaborate_on_last_answer()'s own comment: re-grounds in the
        # exact same retrieved content the question came from, never a
        # fresh search - and simply doesn't match at all (falls through
        # to the branches below, unchanged) whenever the preceding
        # message has no stored context to safely elaborate on.
        elif (
            _is_bare_affirmative_after_question(query)
            and (elaboration := _elaborate_on_last_answer(query))
        ):

            answer, context, source, response_type = elaboration

        # out-of-domain (memory follow-ups always bypass this check).
        # A second, narrower classification decides HOW to respond -
        # never WHETHER to (that's already decided: this branch only
        # runs once is_wlu_related() has already said no). A purely
        # social/emotional message (domain_guard.is_factual_offtopic()
        # -> False) gets a warm, human reaction with no decline
        # phrasing at all, since there's no fact being requested to
        # decline. A genuine factual question about the outside world
        # still gets declined - just in the LLM's own warmer words
        # instead of the flat canned string - never a real answer to
        # the actual question.
        elif (
            normalize_followup_text(query) not in FOLLOWUP_PHRASES
            and not is_wlu_related(query)
        ):

            if is_factual_offtopic(query):
                answer = generate_offtopic_decline(query)
                response_type = "off_topic_decline"
            else:
                answer = generate_offtopic_social_response(query)
                response_type = "off_topic_social"

            context = None
            source = None

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

        # Everything from citation enrichment through message storage is
        # shared with the button-driven follow-up path
        # (_render_followup_buttons(), _finalize_response() - both
        # defined above, near FOLLOWUP_SUGGESTIONS) - see that
        # function's own comment for why passing `answer` through as
        # answer_text matters for citation suppression.
        message = _finalize_response(
            query, answer, source, response_type, st.session_state.memory,
            context
        )

        loading_placeholder.empty()

        with st.chat_message(
            "assistant",
            avatar=ASSISTANT_AVATAR
        ):

            render_response(
                message["response_type"],
                message["content"],
                message["source"],
                message["summary"],
                message["show_card"]
            )

            _render_followup_buttons(
                len(st.session_state.messages) - 1, message
            )

    except Exception as e:

        # Production polish: the user sees a friendly, generic message -
        # raw exception text (e.g. "'NoneType' object has no attribute
        # ...") is confusing and leaks implementation detail. The real
        # exception (with full traceback, not just str(e)) is still
        # printed to the console for whoever's running the app - str(e)
        # alone previously hid exactly which line/call raised, which is
        # what made a real production incident (an exhausted OpenAI
        # account balance) take real investigation to pin down instead
        # of being obvious from console output.
        print(f"Unhandled error while answering {query!r}: {e}")
        traceback.print_exc()

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

        loading_placeholder.empty()

        with st.chat_message(
            "assistant",
            avatar=ASSISTANT_AVATAR
        ):
            st.error(
                error_msg
            )