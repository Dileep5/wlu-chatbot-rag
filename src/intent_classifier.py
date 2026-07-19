import re


def detect_intent(question):

    question = question.lower().strip()

    # =====================================
    # GREETINGS
    # =====================================

    greetings = [
        "hi",
        "hello",
        "hey",
        "hi bro",
        "hello bro",
        "hey bro",
        "hii",
        "heyy",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how r u",
        "what's up",
        "wassup"
    ]

    if question in greetings:
        return "greeting"

    # =====================================
    # THANKS
    # =====================================

    thanks = [
        "thanks",
        "thank you",
        "thankyou",
        "thanks bro",
        "thx"
    ]

    if question in thanks:
        return "thanks"

    # =====================================
    # FOLLOW UP
    # =====================================

    followups = [
        "tell me more",
        "more",
        "explain",
        "details",
        "more details",
        "tell me more about this",
        "what about this",
        "what about it",
        "can you explain"
    ]

    if any(
        phrase in question
        for phrase in followups
    ):
        return "followup"

    # =====================================
    # COURSE CODE
    # =====================================

    if re.search(
        r"\b[A-Z]{2,4}\d{3}[A-Z]?\b",
        question.upper()
    ):
        return "course"

    # =====================================
    # DEPARTMENT
    # =====================================

    department_keywords = [
        "department",
        "faculty",
        "offered by",
        "what programs are offered",
        "physics and computer science",
        "communication studies",
        "history",
        "economics",
        "mathematics"
    ]

    if any(
        keyword in question
        for keyword in department_keywords
    ):
        return "department"

    # =====================================
    # PROGRAM
    # =====================================

    program_keywords = [
        "admission",
        "requirements",
        "master",
        "graduate program",
        "program requirements",
        "coursework",
        "thesis",
        "co-op"
    ]

    if any(
        keyword in question
        for keyword in program_keywords
    ):
        return "program"

    # =====================================
    # DEFAULT
    # =====================================

    return "vector"