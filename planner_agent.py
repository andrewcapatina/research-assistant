# planner_agent.py
from typing import TypedDict, List
from web_scraping import fetch_new_papers
from storage import store_summaries
import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END

from benchmarking import benchmark_summarization_inference
import shared_llm

# llm = shared_llm.get_llm()
llm = shared_llm.get_hf_llm()

class AgentState(TypedDict):
    goal: str
    plan: str
    existing_summaries: List[str]
    new_papers: List[dict]
    new_summaries: List[str]
    final_answer: str

# Node 1: Plan
def plan_step(state):
    prompt = state['goal']
    system_prompt = "Your goal is to transform a high-level research objective into """ + \
        """one (and only one) precise, searchable queries with keywords ONLY that are optimized for academic databases like arXiv. """ + \
        """Include 2-3 key technical terms. No numbering, no explanation."""
    plan = shared_llm.generate(prompt, system=system_prompt, temperature=0.001, max_new_tokens=150)
    print(plan)
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
    check_prompt = \
        f"Goal: {state["goal"]}\n" + \
        "Existing knowledge: {existing}\n" + \
        "Is this sufficient? Answer YES or NO."
    response = shared_llm.generate(check_prompt)
    return {"should_fetch": "NO" in response.upper()}

# Node 4: Fetch New Papers
def fetch_step(state):
    if state.get("should_fetch", True):
        papers = fetch_new_papers(query=state['plan'], days_back=7, max_results=7)
        return {"new_papers": papers}
    return {"new_papers": []}

# Node 5: Summarize New
def summarize_step(state):
    summaries = [""]
    system = "Your task is to read the title and abstract of a research paper and produce a **single paragraph**" + \
        "(3-5 sentences) that captures key findings (what was discovered or achieved), most important technical details" + \
        "(method, architecture, metrics, or innovation), no fluff, no background, no future work."
    for paper in state["new_papers"]:
        summary = benchmark_summarization_inference(paper, system)
        summaries.append(summary)
    store_summaries(state["new_papers"], summaries)
    return {"new_summaries": summaries}

# Node 6: Final Answer
def final_step(state):
    user_prompt = f"""Goal: {state["goal"]}""" + \
            f"""From memory: {state["existing_summaries"]}""" + \
            f"""New findings: {state["new_summaries"]}""" + \
            """Synthesize a final answer."""
    system_prompt = "Analyze and identify 3 key emerging technologies." + \
                    "For each technology, highlight the advancement and challenge/impact."
    answer = shared_llm.summarize(user_prompt, system_prompt, max_tokens=150, temperature=0.5)
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