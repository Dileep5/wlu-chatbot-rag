import os
import re

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

DB_DIR = "data/vector_db"
COLLECTION_NAME = "wlu_chatbot_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

# Increase cutoff slightly for now
DISTANCE_CUTOFF = 1.8

GREETING_PATTERNS = [
    r"^hi$",
    r"^hello$",
    r"^hey$",
    r"^hi bro$",
    r"^hello bro$",
    r"^hey bro$",
    r"^hii+$",
    r"^heyy+$",
    r"^yo$"
]

# -----------------------------
# LOAD EVERYTHING ONLY ONCE
# -----------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(COLLECTION_NAME)

print("System ready.\n")


def is_greeting(text: str) -> bool:
    text = text.lower().strip()

    for pattern in GREETING_PATTERNS:
        if re.fullmatch(pattern, text):
            return True

    return False


def retrieve_top_chunk(query: str):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )

    document = results["documents"][0][0]
    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]

    return document, metadata, distance


def generate_answer(query: str, context: str):

    api_key = os.getenv("OPENAI_API_KEY")

    # No API key yet → fallback mode
    if not api_key:

        return (
            f"Based on the retrieved WLU content:\n\n"
            f"{context[:700]}..."
        )

    # GPT mode
    client = OpenAI()

    prompt = f"""
You are a WLU university assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find that information in the available WLU source."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer only from provided WLU context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def main():

    print("WLU Chatbot")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Greetings
        if is_greeting(query):

            print("Bot: Hello! Ask me anything about WLU.\n")
            continue

        try:

            document, metadata, distance = retrieve_top_chunk(query)

            print(f"\n[DEBUG] Retrieval distance: {distance:.4f}")

            # Weak retrieval
            if distance > DISTANCE_CUTOFF:

                print(
                    "Bot: I could not find that information "
                    "in the available WLU source.\n"
                )

                continue

            answer = generate_answer(query, document)

            print("\nBot:")
            print(answer)

            print("\nSource:")
            print(metadata["url"])

            print()

        except Exception as e:

            print(f"Bot Error: {e}\n")


if __name__ == "__main__":
    main()
