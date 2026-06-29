import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

load_dotenv()

INDEX_PATH = "data/faiss_index"
TOP_K = 5
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_vectorstore():
    if not os.path.exists(INDEX_PATH):
        st.error(
            f"Vector index not found at `{INDEX_PATH}/`. "
            "Please run `python build_index.py` first to build it."
        )
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


HISTORY_WINDOW = 5  # number of past messages to include


def answer_query(vectorstore, query: str, history: list[dict]) -> str:
    results = vectorstore.similarity_search(query, k=TOP_K)
    context = "\n".join([r.page_content for r in results])

    # Build conversation turns from the last HISTORY_WINDOW messages
    history_text = ""
    if history:
        turns = []
        for msg in history[-HISTORY_WINDOW:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{role}: {msg['content']}")
        history_text = "\n".join(turns) + "\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful teacher of Indian history. "
                "Answer ONLY based on the provided context. "
                "Answer suitable for general audience. "
                "If the answer is not present in the context, say \"I don't know\"."
                "You can also translate the answer to different languages if the user asks for it. "
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                + (f"Conversation so far:\n{history_text}\n" if history_text else "")
                + f"User: {query}\nAnswer:"
            ),
        },
    ]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ancient Indian History RAG", page_icon="📜", layout="centered")
st.title("📜 Ancient Indian History Q&A")
st.caption("Ask anything about India's ancient history. Answers are grounded in the source PDF.")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Source:** *India's Ancient History* (full PDF)")
    st.markdown(f"**Model:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBED_MODEL}`")

# ── Load vectorstore from disk (cached) ─────────────────────────────────────
vectorstore = load_vectorstore()

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about ancient Indian history…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = answer_query(vectorstore, user_input, st.session_state.messages[:-1])
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
