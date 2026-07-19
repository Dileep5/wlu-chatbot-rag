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
                    "Wilfrid Laurier University - its programs, courses, "
                    "admissions, tuition, faculty, campus, scholarships, "
                    "student services, or departments.\n\n"
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
