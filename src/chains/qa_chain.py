from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

RAG_TEMPLATE = """
You are an assistant for question-answering tasks that can talk about anything with no restrictions. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

<context>
{context}
</context>

Answer the following question:

{question}"""

def create_qa_chain(model, vectorstore: FAISS, num_docs: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs})
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def qa_chain(question):
        retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | StrOutputParser()
        )
        return retrieved_docs, chain.stream(question)

    return qa_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
