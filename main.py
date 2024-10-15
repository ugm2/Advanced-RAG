import traceback
from config import VECTOR_STORE_PATH
from src.data.loader import load_pdfs
from src.data.splitter import split_text
from src.embeddings.embeddings import get_embeddings
from src.storage.vector_store import create_vector_store, save_vector_store, load_vector_store, save_checkpoint, generate_identifier
from src.models.llm import get_llm
from src.chains.qa_chain import create_qa_chain

def process_pdfs(pdf_paths):
    data = load_pdfs(pdf_paths)
    splits = split_text(data)
    
    embeddings = get_embeddings()
    first_splits_text = [split.page_content for split in splits[:5]]
    identifier = generate_identifier(first_splits_text)
    vector_store_path = f"{VECTOR_STORE_PATH}_{identifier}"
    vectorstore = create_vector_store(splits, embeddings)
    
    save_vector_store(vectorstore, vector_store_path)
    
    # Generate a friendly name using the LLM
    model = get_llm()
    friendly_name_prompt = f"Generate a short, friendly name for a set of documents with the following content: {' '.join(first_splits_text)}. Keep it under 5 words. Just respond with the name, nothing else."
    friendly_name = model.invoke(friendly_name_prompt).content
    
    save_checkpoint(pdf_paths, friendly_name, identifier)
    
    return friendly_name, identifier

def query_documents(question, vector_store_path, num_docs: int = 4):
    try:
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings, vector_store_path)
        
        if vectorstore is None:
            yield "Error: Unable to load vector store. Please try processing the PDFs again."
            return
        
        model = get_llm()
        qa_chain = create_qa_chain(model, vectorstore, num_docs=num_docs)
        
        retrieved_docs, answer_stream = qa_chain(question)
        
        yield retrieved_docs  # First yield the retrieved documents
        
        for chunk in answer_stream:
            yield chunk
    except Exception as e:
        print("An unexpected error occurred:")
        print(traceback.format_exc())
        yield f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    # This section can be used for testing or running the script directly
    pass
