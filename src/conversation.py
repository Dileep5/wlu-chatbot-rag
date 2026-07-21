import re

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
    "tell me a joke",
    "who created you",
    "what are you",
    "can you help me"
]

# Compiled once at module load time. Word-boundary matching (rather than
# plain substring containment) so a short pattern like "hi" only matches
# the standalone word/phrase, not incidentally inside unrelated words
# ("machine", "which", "history") or real faculty names ("White",
# "Okegbile") that happen to contain the same letters.
_CONVERSATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in conversation_patterns) + r")\b"
)


def is_conversation(question):

    question = question.lower().strip()

    return bool(_CONVERSATION_PATTERN.search(question))
