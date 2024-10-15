from langchain_community.document_loaders import PyPDFLoader
from config import PDF_PATH

def load_pdfs(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents
