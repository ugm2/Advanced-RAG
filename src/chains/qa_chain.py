from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

def create_qa_chain(model, vectorstore):
    retriever = vectorstore.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def safe_format_docs(docs):
        try:
            return format_docs(docs)
        except Exception as e:
            print(f"Error in format_docs: {e}")
            return ""

    return lambda question: model.stream(rag_prompt.format(context=safe_format_docs(retriever.invoke(question)), question=question))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
