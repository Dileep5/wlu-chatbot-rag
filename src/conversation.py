# Production Polish Sprint: "tell me a joke" removed - it routed straight
# to generate_chat_response() (app.py), which told an actual joke instead
# of staying WLU-focused. No longer here, it falls through to the normal
# off-topic pipeline (domain_guard.is_wlu_related() -> False ->
# generate_offtopic_decline()), the same warm, WLU-focused redirect every
# other out-of-domain request (weather, sports, trivia) already gets.
conversation_patterns = [

    "how are you",
    "how r u",
    "who are you",
    "what can you do",
    "what did i ask",
    "what did i ask before",
    "do you know me",
    "tell me about yourself",
    "thank you",
    "thanks",
    "okay",
    "ok",
    "great",
    "nice",
    "cool",
    "good",
    "hello",
    "hi",
    "hey",
    "wassup",
    "what's up",
    "who created you",
    "what are you",
    "can you help me"
]

# Trailing/leading punctuation a user might naturally type around one of
# these short patterns ("Hi!", "thanks.", "  hey  ", "what can you do?")
# - stripped before the exact-match check below so punctuation/whitespace
# variation alone never defeats a genuine match.
_CONVERSATION_TRIM_CHARS = " \t\n.,!?;:'\"-"

_CONVERSATION_PATTERNS_SET = set(conversation_patterns)


def is_conversation(question):
    """True only when the message, once lowercased and trimmed of
    surrounding whitespace/punctuation, IS one of conversation_patterns
    outright - not merely CONTAINS one as a substring anywhere.

    Previously used a \\b(?:pattern1|pattern2|...)\\b regex .search(),
    which matches a short pattern ("hey", "hi", "thanks", "ok", ...)
    wherever it appears as a standalone word in the message - including
    as just the first word of an otherwise real, substantial WLU
    question. Confirmed live: "hey can u help me pick electives for
    winter term" matched via the leading "hey" and was routed entirely
    to generate_chat_response() (ungrounded LLM chat, no retrieval, no
    citation) instead of ever reaching hybrid_search(), even though the
    rest of the message is a genuine question. An exact-match check
    after trimming keeps every short-form case working identically
    ("hi", "hey", "thanks", "what can you do", ... alone still match)
    while correctly rejecting any message that merely starts or ends
    with one of these phrases but carries real additional content."""

    normalized = question.lower().strip(_CONVERSATION_TRIM_CHARS)

    return normalized in _CONVERSATION_PATTERNS_SET
