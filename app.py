import streamlit as st
import os
from main import process_pdfs, query_documents

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
            process_pdfs(pdf_paths)
        st.success("PDFs processed successfully!")

    st.subheader("Ask a question")
    question = st.text_input("Enter your question:")
    if question:
        st.write_stream(query_documents(question))

    if st.button("Clear uploaded PDFs"):
        for file_path in pdf_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists("vector_store"):
            import shutil
            shutil.rmtree("vector_store")
        st.rerun()
