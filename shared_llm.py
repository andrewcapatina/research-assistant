# shared_llm.py
from langchain_ollama import OllamaLLM

_llm = None  # Global to hold the instance

OLLAMA_MODEL = "llama3.1:8b"
#OLLAMA_MODEL = "llama3.1:70b"

def get_llm():
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=OLLAMA_MODEL)
    return _llm