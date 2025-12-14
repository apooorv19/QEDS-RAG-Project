import streamlit as st
import os
import pickle
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="QEDS-GPT by Apurva",
    page_icon="✍️",
    layout="wide"
)

# ======================================================================
# CONFIG
# ======================================================================
DB_PATH = "./chroma_db_advanced"
BM25_PATH = "./bm25_index.pkl"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_CONTEXT_CHARS = 5500
CHAT_HISTORY_TURNS = 3
FUSION_VARIANTS = 5

PARAPHRASER_MODEL = "google/flan-t5-small"

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def sanitize_context(text: str) -> str:
    """Clean OCR noise before LLM."""
    text = re.sub(r"<.*?>", " ", text)
    # Remove common OCR artifacts
    text = re.sub(r"(Page\s*\d+|Page\s*No\.?|Saathi|Date\s*\d+/\d+/\d+)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\|\~\^\`]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_semester(query: str):
    match = re.search(r"semester\s*(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def is_vague_query(query: str):
    vague_terms = ["doubt", "doubts", "notes", "semester", "explain"]
    # Only vague if the query is very short AND contains a vague term
    return len(query.split()) < 4 and any(term in query.lower() for term in vague_terms)

def is_context_relevant(query, docs):
    """Simple check to ensure we aren't hallucinating from zero matches."""
    query_tokens = set(query.lower().split())
    # Remove stop words for better matching
    stop_words = {"what", "is", "the", "explain", "how", "to", "in", "of"}
    query_tokens = query_tokens - stop_words
    
    if not query_tokens: return True # Fallback for very short queries
    
    hits = 0
    for d in docs:
        text = d.page_content.lower()
        if any(token in text for token in query_tokens):
            hits += 1
            
    return hits > 0

# ======================================================================
# LOAD RESOURCES
# ======================================================================
@st.cache_resource
def load_paraphraser():
    tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_MODEL)
    return tokenizer, model

tokenizer_flan, model_flan = load_paraphraser()

def generate_paraphrases(query: str, n=FUSION_VARIANTS):
    prompt = f"Generate {n} different paraphrases of: {query}"
    inputs = tokenizer_flan(prompt, return_tensors="pt")
    output = model_flan.generate(
        **inputs,
        max_new_tokens=64,
        num_return_sequences=n,
        do_sample=True,
        top_p=0.95,
        temperature=0.9
    )
    paraphrases = [tokenizer_flan.decode(o, skip_special_tokens=True).strip() for o in output]
    return list(dict.fromkeys(paraphrases))

@st.cache_resource
def load_advanced_pipeline():
    if not os.path.exists(DB_PATH) or not os.path.exists(BM25_PATH):
        st.error("❌ Resources missing. Run ingest.py first.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # Correct usage for updated LangChain libraries
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=reranker
    )

    llm = ChatOllama(model="llama3", temperature=0)

    return ensemble_retriever, reranker, llm

ensemble_retriever, reranker, llm = load_advanced_pipeline()

# ======================================================================
# RAG FUSION RETRIEVAL 
# ======================================================================
def fusion_retrieve(query: str):
    semester = detect_semester(query)
    variants = generate_paraphrases(query)
    variants.append(query)

    all_docs = []

    # 1. Broad Retrieval
    for q in variants:
        if semester:
            docs = ensemble_retriever.invoke(q, filter={"semester": semester})
        else:
            docs = ensemble_retriever.invoke(q)
        if docs:
            all_docs.extend(docs)

    # 2. Deduplicate
    unique_docs = list({d.page_content: d for d in all_docs}.values())
    if not unique_docs:
        return []

    # 3. Smart Reranking with Threshold
    pairs = [[query, d.page_content] for d in unique_docs]
    
    # 🛠️ FIX: Use .score() instead of .predict()
    scores = reranker.model.score(pairs)

    # Combine docs with scores
    scored_docs = list(zip(unique_docs, scores))
    
    # Sort by score (Highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 4. FILTER: Keep only documents with Score > Threshold
    # Threshold -2.0 removes completely irrelevant docs while keeping "okay" matches.
    THRESHOLD = -2.0 
    
    final_docs = []
    for d, score in scored_docs:
        if score > THRESHOLD:
            final_docs.append(d)
    
    # Return Top 3 (or fewer)
    return final_docs[:3]

# ======================================================================
# PROMPT
# ======================================================================

system_template = """
You are an expert Professor of Data Science and Economics acting as a tutor.
Your task is to answer the student's question based strictly on the provided Context (OCR Scanned Notes).

### CRITICAL INSTRUCTIONS:
1. **Invisible Correction:** The Context contains OCR errors (e.g., "S1utsky" instead of "Slutsky", "0ii" instead of $\sigma_{{ii}}$). 
   - You must SILENTLY correct these. 
   - Do NOT say "The text says X but it means Y."
   - Just output the mathematically correct version.
   
2. **Reconstruct Definitions:** If the notes are fragmented, use your expert knowledge to reconstruct the standard definition or theorem implied by the text.

3. **Strict Math Formatting:**
   - ALL math variables and formulas must be in LaTeX.
   - Inline math: $x^2$
   - Block math: $$ E = mc^2 $$
   - Use proper notation (e.g., partial derivatives $\partial$, summations $\sum$).

4. **Tone:** Professional, encouraging, and academic. Do not hallucinate content not supported by the notes or standard theory related to the notes.
"""

human_template = """
CONTEXT (Raw OCR):
{context}

---
CHAT HISTORY:
{chat_history}

STUDENT QUESTION:
{question}
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", human_template)
])

chain = answer_prompt | llm | StrOutputParser()

# ======================================================================
# UI
# ======================================================================
st.title("🎓 QED-Scribe:")
st.caption("Hybrid Search + RAG Fusion + Cross-Encoder Reranker")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a doubt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_resp = ""
        retrieved_sources = set()

        with st.status("🔍 Deep Search in Progress...", expanded=False) as status:
            
            # 1. Vague Query Guard
            if is_vague_query(prompt):
                msg = "Could you please be more specific? I can explain definitions, formulas, or derivations."
                status.update(label="Query too vague", state="error")
                placeholder.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            # 2. Retrieve
            status.update(label="Generating query variants & Searching...")
            docs = fusion_retrieve(prompt)

            if not docs:
                status.update(label="No notes found.", state="error")
                st.warning("I couldn't find any relevant notes in the database.")
                st.stop()

            # 3. Context Check
            if not is_context_relevant(prompt, docs):
                status.update(label="Context irrelevant", state="error")
                msg = "I found some notes, but they don't seem to contain the answer to your specific question."
                placeholder.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.expander("See what I found anyway"):
                    for d in docs: st.text(d.page_content[:200] + "...")
                st.stop()

            # 4. Process Context
            raw_context = "\n\n".join(d.page_content for d in docs)
            context = sanitize_context(raw_context)
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS] + "..."

            # 5. Extract Sources
            for d in docs:
                src = d.metadata.get("source", "Unknown")
                fname = os.path.basename(src)
                clean_name = (fname
                              .replace("_CLEANED.txt", "")
                              .replace(".txt", "")
                              .replace("_", " ")
                              .title())
                retrieved_sources.add(clean_name)

            status.update(label="Knowledge Retrieved & Reranked", state="complete")

        # 6. Generate Answer
        history = [m["content"] for m in st.session_state.messages[-CHAT_HISTORY_TURNS:] if m["role"] == "user"]
        chat_history = "\n".join(history)

        for chunk in chain.stream({
            "context": context,
            "question": prompt,
            "chat_history": chat_history,
        }):
            full_resp += chunk
            placeholder.markdown(full_resp + "▌")

        placeholder.markdown(full_resp)

        # 7. Show Sources
        if retrieved_sources:
            with st.expander("📚 Reference Notes"):
                st.caption("Answer derived from these lecture files:")
                for src in retrieved_sources:
                    st.markdown(f"- 📄 `{src}`")

    st.session_state.messages.append({"role": "assistant", "content": full_resp})
