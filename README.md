# Healthcare RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers healthcare-related questions using information retrieved from PDF documents.

The application includes a built-in healthcare knowledge base and allows users to upload additional PDF documents to expand the knowledge base.

## Features

- Retrieval-Augmented Generation (RAG)
- Built-in healthcare document knowledge base
- Upload one or multiple PDF documents
- Automatic PDF processing
- Text chunking and embedding generation
- Semantic search using ChromaDB
- Local LLM integration with Ollama
- Llama 3.2 model for answer generation
- Source document and page citations
- Relevance score filtering
- Duplicate PDF detection using SHA-256 hashing
- Persistent vector database
- Chat history
- Streamlit web interface

## How It Works

1. PDF documents are loaded using PyPDFLoader.
2. Documents are split into smaller text chunks.
3. Hugging Face sentence-transformer embeddings are generated.
4. Embeddings are stored in ChromaDB.
5. The user's question is converted into an embedding.
6. ChromaDB retrieves the most relevant document chunks.
7. Irrelevant results are filtered using a relevance threshold.
8. The retrieved context and question are sent to Llama 3.2.
9. The model generates an answer based only on the retrieved context.
10. The application displays the answer together with the source documents and page numbers.

## RAG Architecture

User Question

↓

Semantic Search

↓

ChromaDB Vector Database

↓

Relevant Document Chunks

↓

Context + Question

↓

Llama 3.2

↓

Generated Answer

↓

Sources and Page Citations

## Technologies

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Ollama
- Llama 3.2
- PyPDF

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd rag-healthcare-chatbot