"""
PDF
 ↓
Read Text
 ↓
Split into Chunks
 ↓
Create Embeddings
 ↓
Store in ChromaDB
"""
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Paths
PDF_FOLDER = "data/pdfs"
CHROMA_FOLDER = "chroma_db"


# Step 1: Load PDF
loader = PyPDFDirectoryLoader(PDF_FOLDER)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")


# Step 3: Create embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Store chunks in ChromaDB
print("Creating vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_FOLDER
)


print("Success!")
print("Documents were embedded and stored in ChromaDB.")