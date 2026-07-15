import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

CHROMA_FOLDER = "chroma_db"
UPLOAD_FOLDER = "data/uploads"

PROCESSED_HASHES_FILE = "data/processed_hashes.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🩺"
)

st.title("🩺 Healthcare RAG Chatbot")
st.write(
    "Ask questions using the built-in healthcare knowledge base, "
    "or upload additional PDFs to expand it."
)
st.caption(
    "Educational use only. This chatbot does not provide medical diagnosis or treatment."
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm_client():
    hf_token = os.getenv("HF_TOKEN")

    # Local fallback from .streamlit/secrets.toml
    if not hf_token:
        hf_token = st.secrets.get("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN was not found. Add it to local Streamlit secrets "
            "or to the Hugging Face Space secrets."
        )

    return InferenceClient(token=hf_token)

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=load_embeddings()
    )

@st.cache_resource
def load_llm_client():
    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )


vectorstore = load_vectorstore()
llm_client = load_llm_client()

if "active_vectorstore" not in st.session_state:
    st.session_state.active_vectorstore = vectorstore

def process_uploaded_pdfs(uploaded_files, active_vectorstore):
    all_documents = []
    processed_hashes = load_processed_hashes()
    pending_hashes = []
    new_file_count = 0
    skipped_files = []

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)

        if file_hash in processed_hashes:
            skipped_files.append(uploaded_file.name)
            continue

        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        readable_documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not readable_documents:
            raise ValueError(
                f"No readable text was found in {uploaded_file.name}. "
                "The PDF may be scanned or image-based."
            )

        for document in readable_documents:
            document.metadata["uploaded_filename"] = uploaded_file.name
            document.metadata["file_hash"] = file_hash

        all_documents.extend(readable_documents)
        pending_hashes.append(file_hash)
        new_file_count += 1

    if not all_documents:
        return 0, 0, 0, skipped_files

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(all_documents)

    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "No readable text chunks were created from the uploaded PDFs."
        )

    # Add documents first
    active_vectorstore.add_documents(chunks)

    # Save hashes only after successful database insertion
    for file_hash in pending_hashes:
        save_processed_hash(file_hash)

    return (
        new_file_count,
        len(all_documents),
        len(chunks),
        skipped_files
    )
def get_file_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def load_processed_hashes():
    if not os.path.exists(PROCESSED_HASHES_FILE):
        return set()

    with open(PROCESSED_HASHES_FILE, "r", encoding="utf-8") as file:
        return {
            line.strip()
            for line in file
            if line.strip()
        }


def save_processed_hash(file_hash):
    os.makedirs(os.path.dirname(PROCESSED_HASHES_FILE), exist_ok=True)

    with open(PROCESSED_HASHES_FILE, "a", encoding="utf-8") as file:
        file.write(file_hash + "\n")

def create_files_fingerprint(uploaded_files):
    hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(uploaded_file.getvalue())

    return hasher.hexdigest()


st.sidebar.header("Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more healthcare PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if "processed_files_fingerprint" not in st.session_state:
    st.session_state.processed_files_fingerprint = None

if uploaded_files:
    current_fingerprint = create_files_fingerprint(uploaded_files)

    if current_fingerprint != st.session_state.processed_files_fingerprint:
        try:
            with st.sidebar:
                with st.spinner("Adding PDFs to the knowledge base..."):
                    (
                        file_count,
                        page_count,
                        chunk_count,
                        skipped_files
                    ) = process_uploaded_pdfs(
                        uploaded_files,
                        st.session_state.active_vectorstore
                    )

            st.session_state.processed_files_fingerprint = current_fingerprint


            if file_count > 0:
                st.sidebar.success(
                    f"Added {file_count} new PDF files, "
                    f"{page_count} pages, and "
                    f"{chunk_count} chunks."
                )

            if skipped_files:
                st.sidebar.info(
                    "Already in the knowledge base: "
                    + ", ".join(skipped_files)
                )

        except Exception as error:
            st.sidebar.error(
                f"Could not process the PDFs: {error}"
            )
# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['name']} — Page {source['page']}**"
                    )
                    st.write(source["text"])


# Chat input
question = st.chat_input("Ask a question about the healthcare documents")


if question:
    # Save and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant PDF sections
    results_with_scores = (
        st.session_state.active_vectorstore
        .similarity_search_with_relevance_scores(
            question,
            k=4
        )
    )
    st.write(results_with_scores)

    #RELEVANCE_THRESHOLD = 0.4

    results = [
        doc
        for doc, score in results_with_scores
    ]
    # Stop if no relevant information was found
    if not results:
        answer = "I could not find relevant information in the provided document."

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": []
            }
        )

        st.stop()

    context = "\n\n".join(
        [
            f"Source {i + 1}, page "
            f"{doc.metadata.get('page_label', 'Unknown')}:\n"
            f"{doc.page_content}"
            for i, doc in enumerate(results)
        ]
    )

    prompt = f"""
You are a healthcare document assistant.

Answer the user's question using ONLY the document context below.

Strict rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is not clearly stated in the context, say:
  "I could not find the answer in the provided document."
- Give a clear and concise answer in complete sentences.
- Mention the source page when useful.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    with st.chat_message("assistant"):
        with st.spinner("Loading the AI model and searching the documents..."):
            llm_client = load_llm_client()
            response = llm_client.chat_completion(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a healthcare document assistant. "
                            "Use only the provided document context. "
                            "Do not use outside knowledge."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=400
            )

            answer = response.choices[0].message.content

        st.markdown(answer)

        sources = []

        with st.expander("View sources"):
            for i, doc in enumerate(results):
                page = doc.metadata.get("page_label", "Unknown")
                source_name = doc.metadata.get(
                    "uploaded_filename",
                    doc.metadata.get(
                        "title",
                        doc.metadata.get("source", "PDF document")
                    )
                )

                st.markdown(
                    f"**Source {i + 1}: {source_name} — Page {page}**"
                )
                st.write(doc.page_content)

                sources.append(
                    {
                        "name": source_name,
                        "page": page,
                        "text": doc.page_content
                    }
                )

    # Save assistant answer and sources
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )