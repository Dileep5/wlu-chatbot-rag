def is_conversation(question):

    question = question.lower().strip()

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

    return any(
        pattern in question
        for pattern in conversation_patterns
    )