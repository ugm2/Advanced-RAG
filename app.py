import streamlit as st
import os
import shutil
from config import CHECKPOINTS_FILE, NUM_DOCS, VECTOR_STORE_PATH
from main import process_pdfs, query_documents
from src.storage.vector_store import load_checkpoints
from src.chains.qa_chain import format_docs

st.set_page_config(page_title="PDF Q&A System", layout="wide")

st.title("PDF Q&A System")

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    pdf_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
    
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            friendly_name, identifier = process_pdfs(pdf_paths)
        st.success(f"PDFs processed successfully! Checkpoint name: {friendly_name}")
        vector_store_path = f"{VECTOR_STORE_PATH}_{identifier}"
        
        # Reload checkpoints and update the selected checkpoint
        checkpoints = load_checkpoints()
        st.session_state['selected_checkpoint'] = identifier
        st.rerun()  # Rerun the app to refresh the UI

# Move the checkpoint selection after PDF processing
checkpoints = load_checkpoints()
if checkpoints:
    st.subheader("Saved Checkpoints")
    selected_checkpoint = st.selectbox(
        "Select a processed checkpoint:",
        options=list(checkpoints.keys()),
        format_func=lambda x: checkpoints[x]["friendly_name"],
        key='selected_checkpoint'  # Add a key to maintain state
    )
    if selected_checkpoint:
        st.write(f"Using checkpoint: {checkpoints[selected_checkpoint]['friendly_name']}")
        vector_store_path = checkpoints[selected_checkpoint]["vector_store_path"]
    else:
        vector_store_path = None
else:
    vector_store_path = None

st.subheader("Ask a question")
num_docs = st.number_input("Number of fragments to retrieve:", min_value=1, value=NUM_DOCS)
question = st.text_input("Enter your question:")
if question and vector_store_path:
    answer_generator = query_documents(question, vector_store_path, num_docs=num_docs)
    
    try:
        context_docs = next(answer_generator)
        
        with st.expander("Show context", expanded=False):
            for doc in context_docs:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(f"**Page:** {doc.metadata.get('page', 'Unknown') + 1}")
                st.markdown(doc.page_content)
                st.markdown("---")
        
        st.write_stream(answer_generator)
    except StopIteration:
        st.error("Failed to retrieve answer. Please try again.")
elif question:
    st.warning("Please select a checkpoint or process PDFs before asking questions.")

if st.button("Clear all checkpoints"):
    for checkpoint in checkpoints.values():
        if os.path.exists(checkpoint["vector_store_path"]):
            shutil.rmtree(checkpoint["vector_store_path"])
    if os.path.exists(CHECKPOINTS_FILE):
        os.remove(CHECKPOINTS_FILE)
    st.rerun()
