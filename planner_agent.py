from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from web_scraping import fetch_new_papers
from summarize import summarize_paper
from storage import store_summaries
import shared_llm


llm = shared_llm.get_llm()

# planner_agent.py (add this)
def run_planner(goal: str):
    inputs = {"goal": goal}
    result = app.invoke(inputs)
    return result  # Returns final state dict

class AgentState(TypedDict):
    goal: str  # User input, e.g., "Summarize AI hardware papers"
    plan: str  # Generated plan
    papers: List[dict]  # Fetched papers
    summaries: List[str]  # Generated summaries
    evaluation: str  # Final check

# Node 1: Planner
def plan_step(state):
    plan_prompt = PromptTemplate.from_template(
        "Given goal: {goal}\nCreate a step-by-step plan to achieve it."
    )
    plan = llm.invoke(plan_prompt.format(goal=state["goal"]))
    return {"plan": plan}

# Node 2: Fetcher (tool call)
def fetch_step(state):
    # Use plan to customize query, e.g., extract keywords
    papers = fetch_new_papers(query="cat:cs.AR OR cat:cs.LG", max_results=3)  # Limited for demo
    return {"papers": papers}

# Node 3: Summarizer
def summarize_step(state):
    summaries = [summarize_paper(p) for p in state["papers"]]
    store_summaries(state["papers"], summaries)  # Persist
    return {"summaries": summaries}

# Node 4: Evaluator (reasoning check)
def evaluate_step(state):
    eval_prompt = PromptTemplate.from_template(
        "Evaluate summaries for goal {goal}: {summaries}\nIs this complete? Suggest improvements."
    )
    eval = llm.invoke(eval_prompt.format(goal=state["goal"], summaries="\n".join(state["summaries"])))
    return {"evaluation": eval}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("plan", plan_step)
graph.add_node("fetch", fetch_step)
graph.add_node("summarize", summarize_step)
graph.add_node("evaluate", evaluate_step)

# Edges: Sequential for simplicity; add conditionals later (e.g., if no papers, replan)
graph.add_edge("plan", "fetch")
graph.add_edge("fetch", "summarize")
graph.add_edge("summarize", "evaluate")
graph.add_edge("evaluate", END)

graph.set_entry_point("plan")
app = graph.compile()

# Run: inputs = {"goal": "Analyze recent AI hardware papers"}
# result = app.invoke(inputs)
# print(result["evaluation"])