from langchain_community.vectorstores import FAISS
from config import CHECKPOINTS_FILE, VECTOR_STORE_PATH
import os
import json
import hashlib

def create_vector_store(documents, embedding):
    return FAISS.from_documents(documents=documents, embedding=embedding)

def save_vector_store(vectorstore, path=None):
    path = path or VECTOR_STORE_PATH
    vectorstore.save_local(path)
    print(f"Vector store saved successfully to {path}")

def load_vector_store(embedding, path=None):
    vector_store_path = path or VECTOR_STORE_PATH
    if os.path.exists(vector_store_path):
        try:
            return FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("The vector store files might be corrupted. Try reprocessing the PDFs.")
            return None
    else:
        print(f"Vector store not found at {vector_store_path}. Please process PDFs first.")
        return None

def save_checkpoint(pdf_paths, friendly_name, identifier):
    checkpoints = load_checkpoints()
    checkpoints[identifier] = {
        "pdf_paths": pdf_paths,
        "friendly_name": friendly_name,
        "vector_store_path": f"{VECTOR_STORE_PATH}_{identifier}"
    }
    with open(CHECKPOINTS_FILE, "w") as f:
        json.dump(checkpoints, f, indent=4)

def load_checkpoints():
    try:
        with open(CHECKPOINTS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def generate_identifier(splits):
    content = "".join(splits)
    return hashlib.md5(content.encode()).hexdigest()
