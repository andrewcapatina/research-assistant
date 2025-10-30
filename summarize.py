# summarize.py (enhanced with CoT)
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import shared_llm

def summarize_paper(paper, custom_instructions="Summarize focusing on AI innovations:"):
    llm = shared_llm.get_llm()
    cot_prompt = PromptTemplate(
        input_variables=["instructions", "title", "abstract"],
        template="""Step 1: Read the title and abstract carefully.
Step 2: Identify key findings and innovations.
Step 3: Relate to AI/comp arch implications.
Step 4: Output a concise summary.
{instructions}\n\nTitle: {title}\nAbstract: {abstract}"""
    )
    chain = cot_prompt | llm
    response = chain.invoke({"instructions": custom_instructions, "title": paper['title'], "abstract": paper['abstract']})
    return response.strip()