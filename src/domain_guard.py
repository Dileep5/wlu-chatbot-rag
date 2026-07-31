import os
import re

from openai import OpenAI

IN_DOMAIN_KEYWORDS = [
    "wlu",
    "laurier",
    "wilfrid laurier",
    "program",
    "programs",
    "course",
    "courses",
    "admission",
    "admissions",
    "tuition",
    "fees",
    "scholarship",
    "scholarships",
    "financial aid",
    "faculty",
    "faculties",
    "professor",
    "campus",
    "residence",
    "student services",
    "department",
    "departments",
    "credit",
    "credits",
    "undergraduate",
    "graduate",
    "co-op",
    "thesis",
    "registrar",
    # Singular, not "international students" (plural) - a plain
    # substring check means the singular form also matches the plural
    # ("international student" is itself a substring of "international
    # students"), so this one entry covers both phrasings. Confirmed
    # live: "international students" alone missed "im an international
    # student, do i need to renew my study permit" purely because the
    # user's message had no trailing "s".
    "international student",
    "enrolment",
    "enrollment",
    # Campus services / student services / FAQ (Phase 2 corpus expansion
    # added real coverage for these; the keyword list hadn't been
    # extended to match, so legitimate questions about them were falling
    # through to the LLM fallback and, for some phrasings, being
    # misclassified as off-topic before ever reaching retrieval).
    "faq",
    "frequently asked questions",
    "parking",
    "transit",
    "transportation",
    "cycling",
    "onecard",
    "dining",
    "housing",
    "library",
    "classroom",
    "study space",
    "wellness",
    "mental health",
    "counselling",
    "counseling",
    "accessible learning",
    "accessibility",
    "accommodation",
    "indigenous",
    "gendered violence",
    "safety",
    "security",
    "constable",
    "orientation",
    "diversity",
    "equity",
    "immigration",
    # International-student immigration topics specifically (Phase 2's
    # corpus does cover this - only the keyword list was missing these
    # phrasings). Confirmed live: "do i need to renew my study permit"
    # matched none of the existing keywords (not even "immigration",
    # since the phrase never uses that word) and fell through to the
    # LLM classifier, which also misjudged it as off-topic.
    "study permit",
    "immigration status",
    "visa",
    "sustainability",
    "tech services",
    "policy",
    "policies",
    "deadline",
    "deadlines",
    "petition",
    "appeal",
    "academic calendar",
    "important dates",
]

COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")


def matches_wlu_keywords(question: str) -> bool:

    if COURSE_CODE_PATTERN.search(question.upper()):
        return True

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in IN_DOMAIN_KEYWORDS
    )


def classify_with_llm(question: str) -> bool:

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Can't classify without a key - don't block the user.
        return True

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify whether the user's message is about "
                    "Wilfrid Laurier University (WLU) - this includes, but "
                    "is not limited to: its programs, courses, admissions, "
                    "tuition, faculty, campus, scholarships, departments, "
                    "policies, and academic deadlines, as well as ANY "
                    "campus service or student service a university "
                    "typically offers, even if WLU or Laurier isn't named "
                    "explicitly - e.g. parking, transit, dining, "
                    "residence/housing, the library, classrooms and study "
                    "spaces, OneCard, tech services, campus safety/security "
                    "services, wellness and mental health support, "
                    "accessible learning and accommodations, international "
                    "student support, Indigenous student services, "
                    "orientation, diversity and equity resources, and FAQ "
                    "pages for any WLU program or service.\n\n"
                    "Reply with exactly one word: WLU or OFF_TOPIC."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=5
    )

    label = response.choices[0].message.content.strip().upper()

    return label.startswith("WLU")


def is_wlu_related(question: str) -> bool:

    if matches_wlu_keywords(question):
        return True

    return classify_with_llm(question)


def is_factual_offtopic(question: str) -> bool:
    """Only ever called on a message already confirmed off-topic by
    is_wlu_related() above - a second, narrower classification of WHAT
    KIND of off-topic message it is, not whether it's off-topic at all.

    True: a genuine request for a specific fact/piece of information/
    instructions about the outside world (weather, trivia, news, other
    schools, coding help, how-to instructions, general knowledge) -
    app.py must still decline the actual fact for these, just more
    warmly than a flat canned string.

    False: purely social/emotional/conversational content with no real
    information being requested (a greeting, small talk, expressing a
    mood or feeling) - app.py reacts warmly to what the user said
    instead of using decline phrasing at all, since there's no fact
    being asked for and therefore no hallucination risk in reacting to
    it directly.

    Deliberately biased toward True (factual) on any ambiguity: this
    function's only two callers both eventually produce a WLU-focused
    reply either way, but the failure mode of wrongly returning False
    (treating a real question as "just chat") is materially worse than
    the reverse - it risks the warm-chat path producing what looks like
    a real answer to a real question, exactly the hallucination this
    project has spent most of its effort preventing. Wrongly returning
    True for genuinely idle chat only costs a slightly more formal
    decline instead of a warm reaction - a tone miss, not a factual one."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Can't classify without a key - the safer default is to treat
        # it as factual, so the caller declines rather than risks
        # reacting to something that was actually a real question.
        return True

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "The user's message is already known to be unrelated "
                    "to Wilfrid Laurier University. Classify it as one of "
                    "exactly two categories:\n\n"
                    "FACTUAL - the message is a genuine request for a "
                    "specific fact, piece of information, instructions, or "
                    "real-world knowledge - e.g. weather, trivia, news, "
                    "sports results, other schools/companies, how-to "
                    "instructions, coding help, general knowledge "
                    "questions, requests to write or generate something.\n\n"
                    "SOCIAL - the message is purely social, emotional, or "
                    "conversational, with no actual information being "
                    "requested - e.g. a greeting, small talk, expressing a "
                    "mood or feeling, casual chit-chat.\n\n"
                    "If the message mixes both, or you are genuinely "
                    "unsure which it is, classify it as FACTUAL.\n\n"
                    "Reply with exactly one word: FACTUAL or SOCIAL."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=5
    )

    label = response.choices[0].message.content.strip().upper()

    return not label.startswith("SOCIAL")
