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
    "international students",
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
