# summarize.py
import ollama

def summarize_paper(paper, custom_instructions="Summarize the key findings, focusing on implications for computer architecture and AI hardware:"):
    prompt = f"{custom_instructions}\n\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
    response = ollama.generate(model="llama3.1:8b", prompt=prompt)
    return response['response'].strip()  # Ollama returns structured dict