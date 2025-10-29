import argparse
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from web_scraping import fetch_new_papers
from summarize import summarize_paper
from storage import init_db, store_summaries
import json

# Init DB
init_db()

# LCEL setup for orchestration (replaces deprecated LLMChain)
llm = OllamaLLM(model="llama3.1:8b")
prompt_template = PromptTemplate(
    input_variables=["instructions", "title", "abstract"],
    template="{instructions}\n\nTitle: {title}\nAbstract: {abstract}"
)
chain = prompt_template | llm

def run_update(custom_instructions="Summarize focusing on AI innovations:", papers_to_add=None, fetch_auto=True):
    papers = []
    if fetch_auto:
        papers = fetch_new_papers(days_back=7)  # Auto-fetch from arXiv as before
    if papers_to_add:
        papers.extend(papers_to_add)  # Append manual ones
    summaries = []
    for paper in papers:
        # Use chain: 
        response = chain.invoke({"instructions": custom_instructions, "title": paper['title'], "abstract": paper['abstract']})
        summary = response  # OllamaLLM returns a string directly
        summaries.append(summary)
    if papers:  # Only store if there are papers
        store_summaries(papers, summaries)
    return len(papers)

# Parse args for manual adds
parser = argparse.ArgumentParser(description="Run paper summarization")
parser.add_argument('--add-paper', action='append', help='Add a paper: "Title|Abstract|Link|PDF (optional)"')
parser.add_argument('--no-fetch', action='store_true', help='Skip auto-fetch')
parser.add_argument('--manual-file', help='JSON file of papers')

args = parser.parse_args()

# Process manual papers
manual_papers = []
if args.add_paper:
    for p in args.add_paper:
        parts = p.split('|')
        if len(parts) < 3:
            print("Invalid format; use 'Title|Abstract|Link|PDF (optional)'")
            continue
        manual_papers.append({
            'title': parts[0].strip(),
            'abstract': parts[1].strip(),
            'link': parts[2].strip(),
            'pdf': parts[3].strip() if len(parts) > 3 else None,
            'published': None  # No date for manual
        })

if args.manual_file:
    try:
        with open(args.manual_file, 'r') as f:
            manual_papers.extend(json.load(f))
    except FileNotFoundError:
        print(f"File not found: {args.manual_file}")

# Run
run_update(papers_to_add=manual_papers, fetch_auto=not args.no_fetch)