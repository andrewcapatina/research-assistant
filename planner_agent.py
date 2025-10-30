# planner_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from web_scraping import fetch_new_papers
from summarize import summarize_paper
from storage import store_summaries
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import shared_llm

llm = shared_llm.get_llm()

class AgentState(TypedDict):
    goal: str
    plan: str
    existing_summaries: List[str]
    new_papers: List[dict]
    new_summaries: List[str]
    final_answer: str

# Node 1: Plan
def plan_step(state):
    plan_prompt = PromptTemplate.from_template(
        "Goal: {goal}\n"
        "Step 1: Search existing knowledge for relevance.\n"
        "Step 2: If insufficient, fetch new papers from arXiv.\n"
        "Step 3: Summarize and synthesize.\n"
        "Output a concise plan."
    )
    plan = llm.invoke(plan_prompt.format(goal=state["goal"]))
    return {"plan": plan}

# Node 2: Search Memory (FAISS + DB)
def search_memory(state):
    # Reuse your FAISS logic from gui.py
    conn = sqlite3.connect("/app/data/papers.db")
    df = pd.read_sql_query("SELECT summary FROM summaries", conn)
    conn.close()
    if df.empty:
        return {"existing_summaries": []}

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(df['summary'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query_emb = embedder.encode(state["goal"])
    D, I = index.search(np.array([query_emb]), k=5)
    matches = [df.iloc[i]['summary'] for i in I[0] if i != -1]
    return {"existing_summaries": matches}

# Node 3: Decide to Fetch?
def should_fetch(state):
    check_prompt = PromptTemplate.from_template(
        "Goal: {goal}\n"
        "Existing knowledge: {existing}\n"
        "Is this sufficient? Answer YES or NO."
    )
    response = llm.invoke(check_prompt.format(goal=state["goal"], existing="\n".join(state["existing_summaries"])))
    return {"should_fetch": "NO" in response.upper()}

# Node 4: Fetch New Papers
def fetch_step(state):
    if state.get("should_fetch", True):
        papers = fetch_new_papers(days_back=7, max_results=3)
        return {"new_papers": papers}
    return {"new_papers": []}

# Node 5: Summarize New
def summarize_step(state):
    if state["new_papers"]:
        summaries = [summarize_paper(p) for p in state["new_papers"]]
        store_summaries(state["new_papers"], summaries)
        return {"new_summaries": summaries}
    return {"new_summaries": []}

# Node 6: Final Answer
def final_step(state):
    synth_prompt = PromptTemplate.from_template(
        "Goal: {goal}\n"
        "From memory: {memory}\n"
        "New findings: {new}\n"
        "Synthesize a final answer."
    )
    answer = llm.invoke(synth_prompt.format(
        goal=state["goal"],
        memory="\n".join(state["existing_summaries"]),
        new="\n".join(state["new_summaries"])
    ))
    return {"final_answer": answer}

# Build Graph
graph = StateGraph(AgentState)
graph.add_node("plan", plan_step)
graph.add_node("search", search_memory)
graph.add_node("decide", should_fetch)
graph.add_node("fetch", fetch_step)
graph.add_node("summarize", summarize_step)
graph.add_node("answer", final_step)

graph.add_edge("plan", "search")
graph.add_edge("search", "decide")
graph.add_conditional_edges("decide", lambda x: "fetch" if x["should_fetch"] else "answer")
graph.add_edge("fetch", "summarize")
graph.add_edge("summarize", "answer")
graph.add_edge("answer", END)

graph.set_entry_point("plan")
app = graph.compile()

def run_planner(goal: str):
    return app.invoke({"goal": goal})