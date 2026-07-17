# 🩺 Healthcare RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers healthcare questions using trusted PDF documents.

🌐 **Live Demo:** https://rag-healthcare-chatbot.streamlit.app

---

## Overview

The Healthcare RAG Chatbot enables users to ask questions about healthcare topics and receive answers based only on the provided documents.

The chatbot combines semantic search with a Large Language Model (LLM) to generate accurate, document-grounded responses while displaying the source pages used for each answer.

---

## Features

- ✅ Built-in healthcare knowledge base
- ✅ Upload one or more healthcare PDF documents
- ✅ Automatic PDF processing
- ✅ Semantic search using ChromaDB
- ✅ Hugging Face sentence embeddings
- ✅ AI-generated answers using Llama 3.1
- ✅ Source citations with page numbers
- ✅ Multiple PDF support
- ✅ Duplicate upload prevention (per session)
- ✅ Interactive Streamlit interface
- ✅ Cloud deployment

---

## Architecture

```
                    User
                      │
                      ▼
              Streamlit Web App
                      │
                      ▼
          LangChain Retrieval Pipeline
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
  Chroma Vector DB          Hugging Face
 (Semantic Search)         Llama 3.1 Inference
        │
        ▼
 Retrieved PDF Chunks
        │
        ▼
  AI-generated Answer
```

---

## Technologies

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Hugging Face Inference API
- Llama 3.1
- Sentence Transformers
- PyPDF
- Git
- GitHub

---

## Project Structure

```text
Healthcare-RAG-Chatbot/
│
├── app.py
├── app_local.py
├── requirements.txt
├── README.md
├── chroma_db/
├── data/
│   ├── pdfs/
│   └── uploads/
└── .streamlit/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/salehpour1367/rag-healthcare-chatbot.git
```

Go to the project folder:

```bash
cd rag-healthcare-chatbot
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a Streamlit secrets file:

```text
.streamlit/secrets.toml
```

Add your Hugging Face token:

```toml
HF_TOKEN = "your_huggingface_token"
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Questions

- What foods should a person with diabetes eat?
- What is the Diabetes Plate Method?
- Which foods contain carbohydrates?
- What drinks are recommended for people with diabetes?
- What are healthy snack options?

---

## Screenshots

### Home Page

*(Add a screenshot here)*

### Asking a Question

*(Add a screenshot here)*

### Uploading PDFs

*(Add a screenshot here)*

### Source Citations

*(Add a screenshot here)*

---

## Future Improvements

- Conversation memory
- OCR support for scanned PDFs
- Multi-language support
- Medical document summarization
- Citation highlighting
- User authentication

---

## Disclaimer

This application is intended for educational and research purposes only.

It does **not** provide medical diagnosis, treatment, or professional healthcare advice.

Always consult a qualified healthcare professional for medical decisions.

---

## License

MIT License

---

## Author

**Simintaj Salehpour**

GitHub: https://github.com/salehpour1367