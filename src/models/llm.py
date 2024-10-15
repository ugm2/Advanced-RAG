from langchain_ollama import ChatOllama
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K

def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        top_k=LLM_TOP_K
    )
