import os
import re
import streamlit as st
from groq import Groq

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ======================================================
# CONFIG
# ======================================================
DB_PATH = "/app/chroma_db"
COLLECTION_NAME = "semester_notes"

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "llama-3.1-8b-instant"

TOP_K = 3
MAX_CONTEXT_CHARS = 5500
MAX_CHAT_TURNS = 4   # academic memory window


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
# GROQ CLIENT
# ======================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

groq = Groq(api_key=GROQ_API_KEY)


# ======================================================
# SESSION MEMORY (PER USER SESSION)
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# ======================================================
# INTENT CLASSIFICATION
# ======================================================
def is_greeting(query: str) -> bool:
    greetings = {
        "hi", "hello", "hey", "hola",
        "good morning", "good afternoon", "good evening"
    }
    return query.lower().strip() in greetings


def is_meta_query(query: str) -> bool:
    meta_keywords = {
        "user id", "session", "memory", "remember",
        "who are you", "what can you do",
        "how does this work", "about this app",
        "chat history", "clear chat"
    }
    q = query.lower()
    return any(k in q for k in meta_keywords)


def is_vague_query(query: str) -> bool:
    vague_terms = {
        "explain", "doubt", "doubts",
        "topic", "concept", "something", "help"
    }
    tokens = query.lower().split()
    return len(tokens) < 4 and any(t in vague_terms for t in tokens)


# ======================================================
# TEXT SANITIZATION (INTERNAL)
# ======================================================
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
        parts.append(f"Semester {semester}")

    return " – ".join(parts) if parts else "Unknown Source"


@st.cache_data(show_spinner=False)
def beautify_chunk(raw_text: str) -> str:
    prompt = f"""
You are a mathematical editor.

Fix grammar, notation, and equations.
Rewrite equations in clean LaTeX.
Do NOT add new content.
Do NOT mention corrections.

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

if not user_query:
    st.stop()

# Store user message
st.session_state.messages.append({"role": "user", "content": user_query})
with st.chat_message("user"):
    st.markdown(user_query)


# ======================================================
# GREETING HANDLER (NO RAG, NO MEMORY)
# ======================================================
if is_greeting(user_query):
    greeting_prompt = "Respond politely to the greeting."
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": greeting_prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.stop()


# ======================================================
# META HANDLER (NO RAG, NO MEMORY)
# ======================================================
if is_meta_query(user_query):
    meta_prompt = f"""
You are the assistant for the QEDS-GPT application.

Answer clearly and briefly.
Do NOT retrieve notes.
Do NOT mention chat history.
"""
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.stop()


# ======================================================
# VAGUE QUERY GUARD
# ======================================================
if is_vague_query(user_query):
    warning = (
        "Your question is too vague.\n\n"
        "Please ask something specific, for example:\n"
        "- Solve a homogeneous differential equation\n"
        "- Explain weak stationarity\n"
        "- Derive properties of AR(1) process"
    )
    with st.chat_message("assistant"):
        st.markdown(warning)
    st.session_state.messages.append({"role": "assistant", "content": warning})
    st.stop()


# ======================================================
# ACADEMIC RAG (ONLY HERE)
# ======================================================
docs = vectorstore.similarity_search(user_query, k=TOP_K)

if not docs:
    msg = "I could not find relevant notes for this question."
    with st.chat_message("assistant"):
        st.markdown(msg)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.stop()


# Prepare context
raw_context = "\n\n".join(d.page_content for d in docs)
clean_context = sanitize_context(raw_context)[:MAX_CONTEXT_CHARS]

# Academic memory window
chat_history = "\n".join(
    f"{m['role'].upper()}: {m['content']}"
    for m in st.session_state.messages[-MAX_CHAT_TURNS * 2 :]
)

# Sources
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
- Silently correct OCR errors.
- Rewrite equations using LaTeX.
- You MAY use standard academic knowledge.
- Do NOT mention external sources or corrections.
"""


with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        response = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content

    st.markdown(answer)

    st.markdown("**Derived from:**")
    for s in sources:
        st.markdown(f"- {s}")

    st.markdown("---")
    st.markdown("**Relevant Notes:**")
    for i, doc in enumerate(docs, 1):
        with st.expander(f"#{i} — {format_source(doc.metadata)}"):
            st.markdown(beautify_chunk(doc.page_content))


st.session_state.messages.append({"role": "assistant", "content": answer})
