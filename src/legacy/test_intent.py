from intent_classifier import detect_intent

while True:

    q = input("Question: ")

    print(
        detect_intent(q)
    )