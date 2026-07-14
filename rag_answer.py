from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_FOLDER = "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings
)

question = "What foods should a person with diabetes eat?"

results = vectorstore.similarity_search(question, k=3)

print("QUESTION:")
print(question)

print("\nRAG ANSWER BASED ON THE PDF:")
print("According to the retrieved document sections, a person with diabetes should focus on balanced meals, healthy food choices, portion control, and regular meal planning. The PDF suggests learning how different foods affect blood glucose and working with a health care team or dietitian when possible.")

print("\nSOURCES USED:")
for i, doc in enumerate(results):
    print(f"\nSource {i + 1}:")
    print(doc.page_content[:500])
    print("Metadata:", doc.metadata)