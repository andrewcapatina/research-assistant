# shared_llm.py
from langchain_ollama import OllamaLLM

_llm = None  # Global to hold the instance

def get_llm(model="llama3.1:8b"):
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=model)
    return _llm