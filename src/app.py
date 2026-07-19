import os
import re

import streamlit as st
from openai import OpenAI

from retriever import hybrid_search
from conversation import is_conversation

# -----------------------------
# Configuration
# -----------------------------

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
# Helper Functions
# -----------------------------

def is_greeting(text: str) -> bool:

    text = text.lower().strip()

    for pattern in GREETING_PATTERNS:

        if re.fullmatch(pattern, text):
            return True

    return False


def generate_answer(
    query,
    context
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        return (
            "OpenAI API key not found."
        )

    client = OpenAI()

    messages = [

        {
            "role": "system",
            "content": """
You are the official AI assistant
for Wilfrid Laurier University.

You should behave naturally,
similar to ChatGPT or Claude.

Responsibilities:

1. Have natural conversations.
2. Help users with WLU.
3. Answer follow-up questions.
4. Remember previous messages.
5. Use retrieved WLU information
   whenever available.

Rules:

- Be friendly.
- Be conversational.
- Format answers clearly.
- Never invent university facts.
- If information is unavailable,
  clearly say so.
"""
        }

    ]

    messages.extend(
        st.session_state.chat_history[-10:]
    )

    messages.append(
        {
            "role": "user",
            "content": f"""
Retrieved WLU Information:

{context}

Question:

{query}

Answer naturally like ChatGPT.
"""
        }
    )

    response = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return answer


def generate_chat_response(query):

    client = OpenAI()

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",
                "content":
                """
You are a friendly AI assistant
for Wilfrid Laurier University.

Behave naturally like ChatGPT
or Claude.

You may have natural
conversations with users.

Examples:

- how are you
- who are you
- what can you do
- do you know me
- tell me a joke
- what did i ask before

Be friendly and conversational.
                """
            }

        ]

        +

        st.session_state.chat_history[-10:]

        +

        [
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0.8,
        max_tokens=300
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="WLU Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title(
    "Deepu's Sandra WLU Chatbot"
)

st.caption(
    "Hybrid RAG Assistant for "
    "Wilfrid Laurier University"
)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = {
        "last_course": None,
        "last_program": None,
        "last_department": None,
    }


# -----------------------------
# Display Previous Messages
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

        if "source" in msg and msg["source"]:
            st.markdown(
                f"**Source:** {msg['source']}"
            )


# -----------------------------
# User Input
# -----------------------------

query = st.chat_input(
    "Ask something about WLU..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            query
        )

    try:

        # greetings
        if is_greeting(query):

            answer = (
                "Hello! 👋\n\n"
                "How can I help you "
                "with Wilfrid Laurier "
                "University today?"
            )

            source = None

        # normal conversation
        elif is_conversation(query):

            answer = generate_chat_response(
                query
            )

            source = None

        # WLU retrieval
        else:

            context, source = (
                hybrid_search(
                    query,
                    st.session_state.memory
                )
            )

            answer = (
                generate_answer(
                    query,
                    context
                )
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

            if source:
                st.markdown(
                    f"**Source:** "
                    f"{source}"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        error_msg = (
            f"Bot Error: {e}"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_msg
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.error(
                error_msg
            )