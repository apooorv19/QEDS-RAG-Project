import os
import re
import uuid
import streamlit as st
from groq import Groq

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from memory import (
    init_db,
    save_message,
    load_messages,
    clear_user_memory
)

# ======================================================
# CONFIG
# ======================================================
DB_PATH = "/app/chroma_db"
COLLECTION_NAME = "semester_notes"

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama-3.1-8b-instant"

TOP_K = 3
MAX_CONTEXT_CHARS = 5500
MAX_CHAT_TURNS = 4  # memory window


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="QEDS-GPT",
    page_icon="📘",
    layout="wide"
)

st.title("📘 QEDS-GPT")
st.caption("Conversational RAG over 6 semesters of Economics & Data Science notes")


# ======================================================
# INIT MEMORY + USER ID
# ======================================================
init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


# ======================================================
# GROQ CLIENT
# ======================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

groq = Groq(api_key=GROQ_API_KEY)


# ======================================================
# SESSION MEMORY
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(
        st.session_state.user_id,
        limit=MAX_CHAT_TURNS * 2
    )

if st.button("🧹 Clear Chat"):
    clear_user_memory(st.session_state.user_id)
    st.session_state.messages = []
    st.rerun()


# ======================================================
# HELPER FUNCTIONS
# ======================================================
def is_vague_query(query: str) -> bool:
    vague_terms = {
        "explain", "doubt", "doubts", "notes",
        "topic", "concept", "help", "something"
    }
    tokens = query.lower().split()
    return len(tokens) < 4 and any(t in vague_terms for t in tokens)


def sanitize_context(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"(Page\s*\d+|Date\s*\d+/\d+/\d+)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\|\~\^\`]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_source(metadata: dict) -> str:
    subject = metadata.get("subject")
    module = metadata.get("module")
    semester = metadata.get("semester")

    parts = []
    if subject:
        parts.append(subject)
    if module:
        parts.append(module)
    if semester:
        parts.append(f"({semester})")

    return " – ".join(parts) if parts else "Unknown Module"


@st.cache_data(show_spinner=False)
def beautify_chunk(raw_text: str) -> str:
    prompt = f"""
You are a mathematical text editor.

Fix grammar, notation, and equations using LaTeX.
Do NOT add new content.
Do NOT explain corrections.

TEXT:
{raw_text}
"""
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# ======================================================
# LOAD VECTORSTORE
# ======================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )


vectorstore = load_vectorstore()


# ======================================================
# DISPLAY CHAT HISTORY
# ======================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ======================================================
# CHAT INPUT
# ======================================================
user_query = st.chat_input("Ask a question about your notes...")

if user_query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    save_message(st.session_state.user_id, "user", user_query)

    with st.chat_message("user"):
        st.markdown(user_query)

    if is_vague_query(user_query):
        response_text = (
            "Your question is too vague.\n\n"
            "Please ask something specific, for example:\n"
            "- Solve a homogeneous differential equation\n"
            "- Explain weak stationarity\n"
            "- Derive properties of AR(1) process"
        )

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_message(st.session_state.user_id, "assistant", response_text)

        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.stop()

    docs = vectorstore.similarity_search(user_query, k=TOP_K)

    if not docs:
        response_text = "I could not find relevant notes for this question."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_message(st.session_state.user_id, "assistant", response_text)

        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.stop()

    raw_context = "\n\n".join(d.page_content for d in docs)
    clean_context = sanitize_context(raw_context)[:MAX_CONTEXT_CHARS]

    chat_history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages[-MAX_CHAT_TURNS * 2 :]
    )

    sources = list(dict.fromkeys(format_source(d.metadata) for d in docs))

    prompt = f"""
You are an expert professor of Economics and Data Science.

CHAT HISTORY:
{chat_history}

CONTEXT:
{clean_context}

QUESTION:
{user_query}

Rules:
- Silently fix OCR errors
- Rewrite equations in LaTeX
- You may use standard academic knowledge
- Do NOT mention external sources
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = groq.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

        st.markdown(answer)

        st.markdown("**Answer derived mainly from:**")
        for s in sources:
            st.markdown(f"- {s}")

        st.markdown("---")
        st.markdown("**Top Retrieved Chunks:**")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"#{i} — {format_source(doc.metadata)}"):
                st.markdown(beautify_chunk(doc.page_content))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message(st.session_state.user_id, "assistant", answer)
