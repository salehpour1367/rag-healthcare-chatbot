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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🩺"
)

st.title("🩺 Healthcare RAG Chatbot")

st.caption(
    "Educational use only. This chatbot does not provide medical "
    "diagnosis, treatment, or professional medical advice."
)
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:

    st.info(
        """
👋 **Welcome!**

This AI assistant answers healthcare questions using its knowledge base and uploaded PDF documents.

### What you can do

✅ Ask questions about healthcare topics

✅ Upload one or more healthcare PDF files

✅ View the document sources used to generate answers

"""
    )
with st.expander("💡 Example questions"):
    st.markdown("""
- What foods should a person with diabetes eat?
- What is the Diabetes Plate Method?
- Which foods contain carbohydrates?
- What drinks are recommended for people with diabetes?
- What are healthy snack options?
""")

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



vectorstore = load_vectorstore()
llm_client = load_llm_client()

if "active_vectorstore" not in st.session_state:
    st.session_state.active_vectorstore = vectorstore



if "processed_file_hashes" not in st.session_state:
    st.session_state.processed_file_hashes = set()

def initialize_builtin_documents(active_vectorstore):
    # Do nothing if the database already contains documents
    if active_vectorstore._collection.count() > 0:
        return

    pdf_folder = "data/pdfs"

    if not os.path.exists(pdf_folder):
        return

    all_documents = []

    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_folder, filename)
        documents = PyPDFLoader(file_path).load()

        for document in documents:
            document.metadata["uploaded_filename"] = filename

        all_documents.extend(
            document
            for document in documents
            if document.page_content.strip()
        )

    if not all_documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(all_documents)

    if chunks:
        active_vectorstore.add_documents(chunks)
initialize_builtin_documents(
    st.session_state.active_vectorstore
)
def process_uploaded_pdfs(uploaded_files, active_vectorstore):
    all_documents = []
    pending_hashes = []
    new_file_count = 0
    skipped_files = []

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)

        if file_hash in st.session_state.processed_file_hashes:
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

    active_vectorstore.add_documents(chunks)

    for file_hash in pending_hashes:
        st.session_state.processed_file_hashes.add(file_hash)

    return (
        new_file_count,
        len(all_documents),
        len(chunks),
        skipped_files
    )
def get_file_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


st.sidebar.header("📄 Documents")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more healthcare PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.success("✅ Knowledge base ready")

if uploaded_files:
    st.sidebar.markdown("### Uploaded Documents")

    for uploaded_file in uploaded_files:
        st.sidebar.write(f"📄 {uploaded_file.name}")

if uploaded_files:
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

        if file_count > 0:
            st.sidebar.success(
                f"Added {file_count} new PDF files, "
                f"{page_count} pages, and "
                f"{chunk_count} chunks."
            )

        if skipped_files:
            st.sidebar.info(
                "Already processed in this session: "
                + ", ".join(skipped_files)
            )

    except Exception as error:
        st.sidebar.error(
            f"Could not process the PDFs: {error}"
        )



# Create chat history

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**📄 {source['name']} — Page {source['page']}**"
                    )

                    preview = source["text"].strip()

                    if len(preview) > 300:
                        preview = preview[:300] + "..."

                    st.write(preview)

# Download conversation
if st.session_state.messages:
    conversation = (
        "Healthcare RAG Chatbot\n"
        "=========================\n\n"
    )

    for message in st.session_state.messages:

        role = (
            "👤 User"
            if message["role"] == "user"
            else "🤖 Assistant"
        )

        conversation += f"{role}\n"
        conversation += "-" * 40 + "\n"
        conversation += message["content"] + "\n\n"

        if message.get("sources"):

            conversation += "Sources:\n"

            for source in message["sources"]:
                conversation += (
                    f"• {source['name']} "
                    f"(Page {source['page']})\n"
                )

            conversation += "\n"

    conversation += (
        "\n=========================\n"
        "Generated by Healthcare RAG Chatbot\n"
    )
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

    # Retrieve diverse document chunks using MMR
    retriever = st.session_state.active_vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

    retrieved_docs = retriever.invoke(question)

    # Remove duplicate chunks and repeated sources
    results = []
    seen_chunks = set()
    seen_source_pages = set()

    for doc in retrieved_docs:
        source_name = doc.metadata.get(
            "uploaded_filename",
            doc.metadata.get(
                "title",
                doc.metadata.get("source", "PDF document")
            )
        )

        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown")
        )

        normalized_text = " ".join(
            doc.page_content.lower().split()
        )

        chunk_key = (
            source_name,
            str(page),
            normalized_text[:300]
        )

        source_page_key = (
            source_name,
            str(page)
        )

        # Skip exact or near-duplicate chunks
        if chunk_key in seen_chunks:
            continue

        # Show only one chunk from the same file and page
        if source_page_key in seen_source_pages:
            continue

        seen_chunks.add(chunk_key)
        seen_source_pages.add(source_page_key)
        results.append(doc)

        if len(results) == 3:
            break
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
        with st.spinner("Searching the documents and generating an answer..."):
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

        st.markdown("## 🤖 Answer")
        st.markdown(answer)

        sources = []

        st.divider()

        st.markdown("## 📚 Sources")

        for i, doc in enumerate(results):
            page = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "Unknown")
            )

            source_name = doc.metadata.get(
                "uploaded_filename",
                doc.metadata.get(
                    "title",
                    doc.metadata.get("source", "PDF document")
                )
            )

            with st.container(border=True):
                st.markdown(f"**📄 {source_name}**")

                st.caption(f"Page {page}")

                preview = doc.page_content[:250]

                if len(doc.page_content) > 250:
                    preview += "..."

                st.write(preview)

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

st.divider()

st.caption(
    "Healthcare RAG Chatbot • Built with Streamlit, "
    "LangChain, ChromaDB, Hugging Face Embeddings, "
    "and Llama 3.1"
)