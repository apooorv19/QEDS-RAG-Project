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

## 📸 Demo

**Asking about Gini Coefficient:**

![App Demo](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/Demo.png)

**Asking questions about Homogenous Differential Equations:**

![App Demo](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/Demo2.png)
*(The AI retrieves the correct handwritten module, fixes the math symbols, and explains the concept)*

---

## 🚀 Features

### **1️⃣ RAG Fusion**
Generates paraphrased versions of the query using FLAN-T5 and retrieves documents using **multiple query variants**.

### **2️⃣ Hybrid Search**
- BM25 lexical retrieval
- BGE-M3 dense vector retrieval
- Weighted ensemble

### **3️⃣ Cross-Encoder Reranking**
Uses `ms-marco-MiniLM-L-6-v2` to re-rank retrieved chunks for maximum relevance.

### **4️⃣ OCR Noise Handling**
- MathML removal  
- Page number/date stripping  
- Duplicate removal  
- Whitespace normalization  

### **5️⃣ Intelligent Safety Layers**
- Vague-query detection  
- Relevance filtering  
- Semester-based metadata filtering  
- Chat history tracking  

---

## 🧠 Architecture

### **RAG Pipeline**
![Basic RAG Pipeline](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/OCR-RAG%20Architecture.jpg)

### **Detailed Retrieval & Embedding Flow**
![Detailed Architecture](https://github.com/apooorv19/QEDS-RAG-Project/blob/master/assets/RAG-Pipeline.jpg)

```
User Query
↓
FLAN-T5 Paraphraser → {q1, q2, q3, ...}
↓
Hybrid Retrieval (BM25 + BGE-M3 for each qi)
↓
Merged + Deduplicated
↓
Cross-Encoder Reranker
↓
Contextual Compression
↓
OCR Noise Sanitizer
↓
LLaMA3 Response (with LaTeX fixes)
```
### 🧪 Example Capabilities
Query: "Explain the Slutsky substitution effect."

System Action: Retrieves Economics notes from Semester 3, fixes OCR typos in the definition, and presents the derivation.

Query: "Solve the Bernoulli differential equation from Module 1."

System Action: Filters for "Semester 4 - Diff Eq", finds the specific raw formula, converts it to clean LaTeX, and explains the solution steps.

### 🔒 Notes

This repo does not include handwritten text files (private).

Vector DB is ignored (chroma_db_advanced/ not uploaded).

### 📁 Project Structure
```
QED-Scribe/
│
├── 📂 data/
│   ├── 📂 raw_surya_json/      # Output from Surya OCR (Semesters 1-6)
│   └── 📂 ocr_text/            # Cleaned .txt files
│
├── 📂 src/
│   ├── 1_clean_advanced.py     # Regex + LLM Cleaning Pipeline
│   ├── 2_ingest_advanced.py    # Hybrid Ingestion (BGE-M3 + BM25)
│   └── 4_app_advanced.py       # Streamlit RAG Application
│
├── 📂 vector_db/               # ChromaDB storage (GitIgnored)
├── .gitignore                  # Ignores heavy DB files and venv
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

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

3. Run the Cleaning Pipeline
Transforms raw Surya OCR JSON files into clean, readable text files.

```Bash

python src/clean_data.py
```

4. Build the "Brain" (Ingestion)
Generates the Vector Database and Sparse Index.

```Bash

python src/ingest.py
```

5. Launch the Tutor
```Bash

streamlit run src/app.py
```

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
