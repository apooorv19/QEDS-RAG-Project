# QEDS-GPT - Hybrid RAG Academic Assistant

QEDS-GPT is a modular Hybrid Retrieval-Augmented Generation application for answering questions from Economics, Statistics, Mathematics, Data Science, Machine Learning, and semester-note content.

The project combines Streamlit, ChromaDB, HuggingFace embeddings, BM25 keyword retrieval, Reciprocal Rank Fusion, conversation memory, and Groq LLM generation into a clean academic chatbot.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![LangChain](https://img.shields.io/badge/RAG-LangChain-orange.svg)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq-black.svg)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue.svg)

---

## Overview

QEDS-GPT lets users ask natural-language academic questions and receive concise, structured answers. For academic queries, the app retrieves relevant note chunks from a persisted ChromaDB database, combines semantic and keyword search results, builds a grounded prompt, and generates an answer using Groq.

The app also supports conversational follow-ups, semester filtering, retrieved source display, and general-knowledge fallback when the answer is not found in the notes.

---

## Key Features

- **Hybrid Retrieval:** combines dense vector retrieval with BM25 keyword retrieval.
- **Reciprocal Rank Fusion:** merges dense and BM25 rankings without comparing incompatible raw scores.
- **ChromaDB Vector Store:** stores embedded academic note chunks and metadata.
- **HuggingFace Embeddings:** uses `BAAI/bge-m3` for dense semantic search.
- **Groq LLM:** uses `llama-3.1-8b-instant` for classification, summarization, and answer generation.
- **Query Classification:** routes questions as academic, greeting, meta, general, or prompt-injection attempts.
- **Conversation Memory:** stores recent chat turns using Streamlit session state.
- **Conversation Summarization:** summarizes older context after a threshold.
- **Semester Filtering:** allows retrieval filtering by semester.
- **Source Display:** shows sources and retrieved chunks only when note context is used.
- **General-Knowledge Fallback:** answers from model knowledge when notes do not contain the topic, without displaying false citations.
- **Docker Support:** includes a Dockerfile for reproducible local/container runs.

---

## Architecture

```text
User
  |
  v
Streamlit Chat UI
  |
  v
Chat Memory + Query Classifier
  |
  +--> Greeting / Meta / General / Injection response
  |
  +--> Academic RAG pipeline
          |
          v
      Hybrid Retriever
          |
          +--> Dense Retrieval with ChromaDB + HuggingFace Embeddings
          |
          +--> BM25 Keyword Retrieval
          |
          v
      Reciprocal Rank Fusion
          |
          v
      Context Cleaning + Prompt Builder
          |
          v
      Groq LLM
          |
          v
      Answer + Sources + Memory Update
```

---

## How the Application Works

1. The user enters a question in the Streamlit chat input.
2. Recent conversation history is loaded from Streamlit session state.
3. The query classifier categorizes the question.
4. Non-academic questions are handled directly.
5. Academic questions trigger retrieval.
6. Dense retrieval searches ChromaDB using HuggingFace embeddings.
7. BM25 performs keyword retrieval over the same document chunks.
8. Reciprocal Rank Fusion combines both ranked lists.
9. Optional semester filtering is applied.
10. Top chunks are cleaned and placed into the prompt.
11. Groq generates the final answer.
12. The UI displays the answer, sources, retrieved chunks, and response time.
13. The conversation memory is updated.

---

## Project Structure

```text
qeds-gpt-refactored-hf/
│
├── chroma_db/
│   └── Persisted ChromaDB vector database files
│
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── config.py
│   ├── llm.py
│   ├── memory.py
│   ├── prompts.py
│   ├── retriever.py
│   ├── services.py
│   ├── streamlit_app.py
│   ├── text_processor.py
│   └── ui.py
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Module Guide

| File | Purpose |
|---|---|
| `src/streamlit_app.py` | Main Streamlit entrypoint. Coordinates UI rendering, chat input, query routing, and service initialization. |
| `src/config.py` | Central configuration for model names, database path, retrieval settings, memory settings, and query categories. |
| `src/prompts.py` | Stores academic, classifier, summary, meta, and user prompt templates. |
| `src/retriever.py` | Loads ChromaDB, HuggingFace embeddings, BM25 retriever, vector retriever, and applies RRF. |
| `src/classifier.py` | Classifies user queries into supported categories using the LLM. |
| `src/llm.py` | Wraps Groq API usage and handles LLM errors. |
| `src/memory.py` | Manages Streamlit session-state chat memory and summarization. |
| `src/services.py` | Business logic layer that coordinates LLM, classifier, retriever, prompts, and response generation. |
| `src/ui.py` | Contains Streamlit UI helper functions for sidebar, chat history, answer display, sources, and status messages. |
| `src/text_processor.py` | Cleans retrieved text and formats document source metadata. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit |
| RAG tooling | LangChain |
| Vector database | ChromaDB |
| Embeddings | HuggingFace `BAAI/bge-m3` |
| Sparse retrieval | BM25 |
| Fusion method | Reciprocal Rank Fusion |
| LLM provider | Groq |
| Default LLM | `llama-3.1-8b-instant` |
| Containerization | Docker |
| Large file handling | Git LFS for `chroma_db/` |

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

For local development:

```bash
pip install -r requirements-local.txt
```

For Docker/container deployment:

```bash
pip install -r requirements.txt
```

### 4. Add Your Groq API Key

Create this local file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Do not commit this file to GitHub.

### 5. Run the App

```bash
streamlit run src/streamlit_app.py
```

---

## Docker Usage

Build the Docker image:

```bash
docker build -t qeds-gpt .
```

Run the container:

```bash
docker run -p 7860:7860 -e GROQ_API_KEY=your_groq_api_key_here qeds-gpt
```

Open:

```text
http://localhost:7860
```

The Dockerfile runs:

```bash
streamlit run src/streamlit_app.py --server.port=7860 --server.address=0.0.0.0
```
---

## Troubleshooting

### `GROQ_API_KEY not configured`

Create `.streamlit/secrets.toml` locally or pass the key as an environment variable:

```bash
set GROQ_API_KEY=your_groq_api_key_here
```

On macOS/Linux:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### ChromaDB Is Not Found

Confirm this directory exists:

```text
chroma_db/
```

If your database is somewhere else, set:

```bash
CHROMA_DB_PATH=path/to/chroma_db
```

### First Academic Question Is Slow

The first academic query may initialize:

- HuggingFace embedding model
- ChromaDB
- BM25 index

This can take time, especially on CPU.

### GitHub Rejects Large Files

Use Git LFS:

```bash
git lfs install
git lfs track "chroma_db/**"
```

---

## Future Improvements

- Add automated tests.
- Add retrieval quality evaluation.
- Add streaming LLM responses.
- Add optional reranking.
- Add authentication for private deployments.
- Add persistent user accounts and chat history.
- Add observability for latency and retrieval diagnostics.

---

## Author

**Apurva Mishra**  
IMSc Quantitative Economics and Data Science  
Birla Institute of Technology, Mesra

GitHub: [apooorv19](https://github.com/apooorv19)  
LinkedIn: [Apurva Mishra](https://www.linkedin.com/in/apooorv/)
