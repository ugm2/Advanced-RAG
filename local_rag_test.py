from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the PDF document
pdf_path = "national_ai_rd_strategic_plan.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()
print(f"Number of pages loaded from PDF: {len(data)}")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print(f"Number of text chunks after splitting: {len(all_splits)}")

# Create embeddings using Ollama
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Generating embeddings for document chunks...")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=local_embeddings)
print("Embeddings generated and vector store created successfully.")

# Perform similarity search on the vector store
question = "What are the main topics covered in this document?"
print(f"Performing similarity search for question: '{question}'")
docs = vectorstore.similarity_search(question)
print(f"Number of relevant document chunks retrieved: {len(docs)}")
if docs:
    print("First retrieved document chunk:")
    print(docs[0])
else:
    print("No relevant document chunks found.")

# Set up the LLM model with Ollama
model = ChatOllama(model="llama3.2:3b")
print("LLM model initialized with 'llama3.2:3b'.")

# Summarize the retrieved documents
prompt = ChatPromptTemplate.from_template("Summarize the main themes in these retrieved docs: {docs}")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = {"docs": format_docs} | prompt | model | StrOutputParser()
print("Generating summary of the retrieved document chunks...")
chain_response = chain.invoke(docs)
print("Summary of the main themes:")
print(chain_response)

# Question Answering using retrieved docs
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

qa_chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "What is the desired outcome of the NAI RD Strategic Plan?"
print(f"Performing Q&A for question: '{question}'")
qa_response = qa_chain.invoke({"context": docs, "question": question})
print("Response to the question:")
print(qa_response)

# Automatic Q&A with retrieval
retriever = vectorstore.as_retriever()
qa_chain_auto = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

print("Performing automatic Q&A with retrieval...")
qa_auto_response = qa_chain_auto.invoke(question)
print("Automatic Q&A response:")
print(qa_auto_response)

# Save vectorstore for later use
print("Saving vector store for future use...")
# vectorstore.save_local("vector_store_path")
print("Vector store saved successfully.")