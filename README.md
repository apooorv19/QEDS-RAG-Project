# 🎓 QED-Scribe - Hybrid RAG

This project is a **production-grade Retrieval-Augmented Generation (RAG) Tutor** built to understand and explain **handwritten notes of QEDS**, even with OCR noise.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Stack](https://img.shields.io/badge/Stack-LangChain_|_Streamlit_|_Ollama-orange.svg)
![OCR](https://img.shields.io/badge/OCR-Surya-green.svg)

It uses:

- 🔍 **Hybrid Retrieval (BM25 + BGE-M3)**
- 🔄 **RAG Fusion** using FLAN-T5 paraphraser
- 🔥 **Cross-Encoder Reranking**
- ✂️ **Contextual Compression**
- 🧹 **OCR Noise Sanitization**
- 🧠 **LLaMA3 (Ollama)** for answering

Perfect for academic notes, handwritten documents, mathematical derivations, and noisy OCR text.

---

## 📸 Demo

**Asking about Gini Coefficient:**

![App Demo](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/Demo.png)

**Asking questions about Homogenous Differential Equations:**

![App Demo](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/Demo2.png)
*(The AI retrieves the correct handwritten module, fixes the math symbols, and explains the concept)*

---

## 🔍 What Problem Does This Solve?

Students often struggle with:
- Handwritten lecture notes
- OCR errors in scanned material
- Fragmented definitions and formulas
- Losing context while asking follow-up questions

**QEDS-GPT solves this by:**
- Indexing cleaned OCR notes into a semantic vector database
- Retrieving the most relevant modules and topics
- Correcting equations and notations
- Allowing multi-turn, memory-aware academic conversations

## 🚀 Features

### 🧠 Conversational RAG with Memory
- Multi-turn chat interface
- Context retained across refreshes and sessions
- No login required

### 📚 Semantic Retrieval over Notes
- ChromaDB vector store
- Hugging Face `bge-m3` embeddings
- Module-aware retrieval with metadata:
  - Semester
  - Subject
  - Module

### ✍️ OCR Noise Correction
- Fixes broken equations and symbols
- Rewrites math in clean **LaTeX**
- Repairs fragmented sentences

### 🚫 Hallucination Controls
- Vague-query detection
- Relevance filtering
- Uses academic knowledge only to *reconstruct* missing context

### 📦 Production Deployment
- Dockerized application
- Deployed on **Hugging Face Spaces**
- Git LFS support for vector index files
- Secure API key management

---

## 🧠 Architecture

### **RAG Pipeline**
![Basic RAG Pipeline](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/OCR-RAG%20Architecture.jpg)

### **Detailed Retrieval & Embedding Flow**
![Detailed Architecture](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/RAG-Pipeline.jpg)

```
User
↓
Streamlit Chat UI
↓
Semantic Retrieval (ChromaDB)
↓
OCR Cleanup & Context Sanitization
↓
LLM Reasoning (Groq)
↓
Answer + Updated Memory
```
---
### 🧪 Example Capabilities
Query: "Explain the Slutsky substitution effect."

System Action: Retrieves Economics notes from Semester 3, fixes OCR typos in the definition, and presents the derivation.

Query: "Solve the Bernoulli differential equation from Module 1."

System Action: Filters for "Semester 4 - Diff Eq", finds the specific raw formula, converts it to clean LaTeX, and explains the solution steps.

### 🔒 Notes

Handwritten notes are private and not included

Vector database is stored using Git LFS

Memory is stored per user locally via SQLite

---

### 📁 Project Structure
```
QEDS-RAG-Project/
│
├── chroma_db/              # Vector database
│
├── src/
│ ├── streamlit_app.py      # Main conversational RAG app   
│
├── Dockerfile
├── requirements.txt
└── README.md
```
---
### ⚙️ Tech Stack

```
- Language: Python
- UI: Streamlit
- Vector DB: ChromaDB
- Embeddings: Hugging Face `BAAI/bge-m3`
- LLM: Groq (LLaMA-3.1-8B-Instant)
- Deployment: Docker + Hugging Face Spaces
```
---
### 🛠️ Installation & Usage

#### 1. Prerequisites  
Make sure Ollama is installed and running with Llama 3:

```bash
ollama run llama3
```

#### 2. Setup Environment

```Bash

git clone https://github.com/apooorv19/QEDS-RAG-Project.git
cd QEDS-RAG-Project
pip install -r requirements.txt
```

3. Set environment variable

```Bash
export GROQ_API_KEY=your_api_key_here
```

4. Run the app
```Bash
streamlit run src/streamlit_app.py
```

### 🐳 Docker Deployment
```Bash
docker build -t qeds-gpt .
docker run -p 8501:8501 -e GROQ_API_KEY=your_api_key qeds-gpt
```

### 🌐 Live Demo
https://huggingface.co/spaces/Apooorv69/QEDS-RAG-Project

---

### 👤 Author

Apurva Mishra <br>
IMSc Quantitative Economics & Data Science <br>
Birla Institute of Technology, Mesra <br>

GitHub: https://github.com/apooorv19
<br>
LinkedIn: https://www.linkedin.com/in/apooorv/

---

### 📜 Citations & Credits
```
@misc{paruchuri2025surya,
  author       = {Vikas Paruchuri and Datalab Team},
  title        = {Surya: A lightweight document OCR and analysis toolkit},
  year         = {2025},
  howpublished = {\url{https://github.com/VikParuchuri/surya}},
  note         = {GitHub repository},
}
```
