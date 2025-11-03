# ==== summarize.py (fixed) ====
import shared_llm
import re

def _clean_summary(raw: str) -> str:
    """Strip CoT, prompt, and keep only the final answer."""
    # Remove everything up to the first blank line after the abstract
    parts = re.split(r"\n\s*\n", raw, 1)
    candidate = parts[-1] if len(parts) > 1 else raw

    # Keep only the first 2-3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", candidate)
    return " ".join(sentences[:3]).strip()

def summarize_paper(paper, custom_instructions="Summarize focusing on electrical and computer engineering innovations:"):
    llm = shared_llm.get_hf_llm()

    # 1. Minimal prompt – no CoT steps
    prompt = PromptTemplate.from_template(
        "Title: {title}\n"
        "Abstract: {abstract}\n\n"
        "{instructions}\n"
        "Provide a concise 2-3 sentence summary. Do not repeat any input."
    )

    chain = prompt | llm
    raw = chain.invoke({
        "title": paper["title"],
        "abstract": paper["abstract"],
        "instructions": custom_instructions,
    })
    print(raw)
    return _clean_summary(raw)