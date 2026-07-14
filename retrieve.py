from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Path to the existing vector database
CHROMA_FOLDER = "chroma_db"


# Step 1: Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 2: Load the existing ChromaDB
vectorstore = Chroma(
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings
)


# Step 3: Ask a question
question = "What foods should a person with diabetes eat?"

print(f"Question: {question}")


# Step 4: Search for the 3 most relevant chunks
results = vectorstore.similarity_search(
    question,
    k=3
)


# Step 5: Display the results
for i, doc in enumerate(results):
    print("\n" + "=" * 50)
    print(f"RESULT {i + 1}")
    print("=" * 50)

    print(doc.page_content)

    print("\nSource:")
    print(doc.metadata)