from langchain_community.vectorstores import FAISS
from config import VECTOR_STORE_PATH
import os

def create_vector_store(documents, embedding):
    return FAISS.from_documents(documents=documents, embedding=embedding)

def save_vector_store(vectorstore):
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved successfully to {VECTOR_STORE_PATH}")

def load_vector_store(embedding):
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            return FAISS.load_local(VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("The vector store files might be corrupted. Try reprocessing the PDFs.")
            return None
    else:
        print("Vector store not found. Please process PDFs first.")
        return None
