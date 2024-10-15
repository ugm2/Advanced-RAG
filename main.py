import traceback
from src.data.loader import load_pdfs
from src.data.splitter import split_text
from src.embeddings.embeddings import get_embeddings
from src.storage.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.models.llm import get_llm
from src.chains.qa_chain import create_qa_chain

def process_pdfs(pdf_paths):
    data = load_pdfs(pdf_paths)
    splits = split_text(data)
    
    embeddings = get_embeddings()
    vectorstore = create_vector_store(splits, embeddings)
    
    save_vector_store(vectorstore)

def query_documents(question):
    try:
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings)
        
        if vectorstore is None:
            yield "Error: Unable to load vector store. Please try processing the PDFs again."
            return
        
        model = get_llm()
        qa_chain = create_qa_chain(model, vectorstore)
        
        for chunk in qa_chain(question):
            yield chunk
    except Exception as e:
        print("An unexpected error occurred:")
        print(traceback.format_exc())
        yield f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    # This section can be used for testing or running the script directly
    pass
