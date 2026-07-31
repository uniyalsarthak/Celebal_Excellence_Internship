# Smart Research Assistant (RAG-Based Knowledge System)

## Overview

Smart Research Assistant is a Retrieval-Augmented Generation (RAG) application that enables users to interact with PDF documents using natural language. Instead of manually searching through lengthy documents, users can upload a PDF and ask questions to receive accurate, context-aware responses grounded in the document.


---

## Features

- Upload PDF documents
- Intelligent document chunking
- Semantic search using vector embeddings
- Source-grounded responses
- Supports both FAISS and ChromaDB vector stores
- Automated evaluation pipeline
- Streamlit-based interactive interface

---

## Tech Stack

- Python
- LangChain
- Groq (Llama 3.3)
- HuggingFace Embeddings
- FAISS
- ChromaDB
- PyMuPDF
- Streamlit

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Smart_Hiring_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```text
GROQ_API_KEY=YOUR_GROQ_API_KEY
MODEL_NAME=llama-3.3-70b-versatile
```

### 5. Run the application

```bash
streamlit run app.py
```

### 6. Upload a PDF

The application will automatically:

- Extract text
- Generate embeddings
- Build the FAISS index
- Start answering questions