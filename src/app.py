import os
import re

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
DB_DIR = "data/vector_db"
COLLECTION_NAME = "wlu_chatbot_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
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
    r"^yo$",
    r"^good morning$",
    r"^good afternoon$",
    r"^good evening$",
]


# -----------------------------
# Helper functions
# -----------------------------
def is_greeting(text: str) -> bool:
    text = text.lower().strip()
    for pattern in GREETING_PATTERNS:
        if re.fullmatch(pattern, text):
            return True
    return False


@st.cache_resource
def load_resources():
    """Load model and ChromaDB only once."""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve_top_chunk(query: str, model, collection):
    """Return the most relevant chunk and its metadata."""
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
    """
    If OpenAI key exists, use GPT.
    Otherwise return a simple context-based fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return f"Based on the retrieved WLU content:\n\n{context[:700]}..."

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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="WLU Chatbot", page_icon="🎓", layout="centered")

st.title("Deepu's Sandra WLU Chatbot")
st.caption("A simple RAG-based university information assistant")

model, collection = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about WLU...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Handle greetings
    if is_greeting(query):
        answer = "Hello! Ask me anything about WLU."
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

    else:
        try:
            document, metadata, distance = retrieve_top_chunk(query, model, collection)

            with st.chat_message("assistant"):
                st.write(f"[DEBUG] Retrieval distance: {distance:.4f}")

                if distance > DISTANCE_CUTOFF:
                    answer = "I could not find that information in the available WLU source."
                    st.markdown(answer)
                else:
                    answer = generate_answer(query, document)
                    st.markdown(answer)
                    st.markdown(f"**Source:** {metadata.get('url', 'No source found')}")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = f"Bot error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)