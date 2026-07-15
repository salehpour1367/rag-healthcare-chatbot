---
title: Healthcare RAG Chatbot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---
# 🩺 Healthcare RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers healthcare-related questions using PDF documents.

The chatbot combines semantic search with a Large Language Model (LLM) to retrieve relevant information from healthcare PDFs and generate accurate, context-based answers.

---

## Features

- 📄 Ask questions about healthcare PDF documents
- 📚 Built-in healthcare knowledge base
- ➕ Upload one or multiple PDF files
- 🔍 Semantic search using vector embeddings
- 🤖 Llama 3.2 (Ollama) integration
- 💬 Chat interface built with Streamlit
- 📑 Displays the source document and page number
- 🚫 Prevents duplicate PDF uploads
- ⚡ Automatically processes newly uploaded PDFs
- 💾 Persistent Chroma vector database
- 🧠 Context-aware Retrieval-Augmented Generation (RAG)

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Ollama
- Llama 3.2
- PyPDFLoader

---

## Project Structure

```
rag-healthcare-chatbot/
│
├── app.py                  # Streamlit application
├── ingest.py               # Build the initial knowledge base
├── retrieve.py             # Retrieve relevant documents
├── rag_answer.py           # Generate answers
├── requirements.txt
│
├── chroma_db/              # Persistent vector database
├── data/
│   ├── pdfs/               # Built-in healthcare PDFs
│   ├── uploads/            # User uploaded PDFs
│   └── processed_hashes.txt
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/salehpour1367/rag-healthcare-chatbot.git
cd rag-healthcare-chatbot
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it (Windows)

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install Ollama and download the model

```bash
ollama pull llama3.2:3b
```

Run the application

```bash
streamlit run app.py
```

---

## How It Works

1. Load the built-in healthcare knowledge base.
2. Upload one or more healthcare PDF documents.
3. PDFs are automatically processed and converted into embeddings.
4. Chunks are stored in ChromaDB.
5. User questions are converted into embeddings.
6. The most relevant chunks are retrieved.
7. Llama 3.2 generates an answer using only the retrieved context.
8. The chatbot displays both the answer and the supporting sources.

---

## Future Improvements

- Support DOCX and TXT documents
- Conversation memory
- Medical citation formatting
- Cloud deployment
- User authentication
- Multi-language support

---

## License

This project is licensed under the MIT License.