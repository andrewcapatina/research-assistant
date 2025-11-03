import streamlit as st
import sqlite3
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import shared_llm
import planner_agent
import storage

if "planner_reload_flag" not in st.session_state:
    st.session_state.planner_reload_flag = False
if "planner_results" not in st.session_state:
    st.session_state.planner_results = False    
planner_reload_flag = st.session_state.planner_reload_flag
planner_results = st.session_state.planner_results

# Build FAISS index for semantic search (from all summaries in DB)
@st.cache_resource  # Cache for performance
def build_faiss_index():
    conn = sqlite3.connect(storage.DB_PATH)
    df_all = pd.read_sql_query("SELECT id, paper_json, summary FROM summaries", conn)

    conn.close()
    
    if df_all.empty:
        return None, None
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(df_all['summary'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(np.array(embeddings))
    return index, df_all  # Return index and DF for ID mapping

# Streamlit app
st.title("Research Paper Summaries Dashboard")

# Filters for standard search
weeks = st.slider("Look back weeks:", 1, 4, 1)
keyword = st.text_input("Keyword search (in title/abstract/summary):")

df = storage.get_summaries(weeks_back=weeks, keyword=keyword)

if not df.empty:
    # Optional: Synthesize if checkbox selected
    if st.checkbox("Synthesize into report (via Ollama)"):
        summaries = df['summary'].tolist()
        combined = "\n\n".join(summaries)
        synth_prompt = f"Compile these distinct paper summaries into a unique, non-repetitive report. Structure as: 1. Overview of key themes. 2. Bullet points of major innovations per paper (no duplicates). 3. Implications for AI/comp arch. Be concise, avoid filler:\n\n{combined[:3000]}"
        with st.spinner("Synthesizing..."):
            llm = shared_llm.get_hf_llm()
            response = llm.invoke(synth_prompt)
        st.markdown("### Synthesized Report")
        st.write(response)

# New: Semantic Search Section for Memory/Reasoning
st.markdown("### Semantic Search (Query Past Summaries)")
semantic_query = st.text_input("Semantic query (e.g., 'AI hardware innovations'):")
if st.button("Search Memory") and semantic_query:
    index, df_all = build_faiss_index()
    if index is None:
        st.info("No summaries in memory yet.")
    else:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = embedder.encode(semantic_query)
        distances, indices = index.search(np.array([query_emb]), k=5)  # Top 5 matches
        st.markdown("### Top Matching Summaries")
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue  # No match
            row = df_all.iloc[idx]
            st.write(f"**Match {i+1} (Score: {distances[0][i]:.2f}):** {row['summary'][:200]}...")
            # Optionally fetch full paper details from DB using row['id']

# New: Goal-Oriented Planner Section
st.markdown("### Agentic Planner: Set a Goal")
goal_input = st.text_input("Enter a goal (e.g., 'Analyze recent AI hardware papers'):")
if st.button("Run Planner") and goal_input:
    planner_results = None
    with st.spinner("Planning and executing..."):
        planner_results = planner_agent.run_planner(goal_input)
        st.session_state.planner_results = planner_results
        planner_reload_flag = True
        st.session_state.planner_reload_flag = planner_reload_flag
        st.rerun()

if planner_reload_flag is True:
    st.markdown("### Planner Results")
    # Display structured output
    with st.expander("Generated Plan"):
        st.write(planner_results.get("plan", "No plan generated."))

    summaries = planner_results.get("summaries", [])
    if summaries:
        st.markdown("### Summaries")
        for i, summary in enumerate(summaries, 1):
            st.write(f"**Summary {i}:** {summary}")

    with st.expander("Evaluation"):
        st.write(planner_results.get("final_answer", "No evaluation generated."))
 
    st.session_state.planner_reload_flag = planner_reload_flag

st.markdown("### Database Management")
if st.button("Clear Database"):
    storage.clear_db()
    planner_reload_flag = False
    st.rerun()

if not df.empty:
    # Display table
    st.markdown("### Paper Summaries")
    for _, row in df.iterrows():
        paper = pd.read_json(StringIO(row['paper_json']), typ='series')  # Parse JSON
        with st.expander(f"{paper['title']} ({row['date'] or 'Manual'})"):
            st.markdown(f"**Link:** [{paper['link']}]({paper['link']})")
            st.markdown(f"**Abstract:** {paper['abstract'][:500]}...")  # Truncate
            st.markdown("**Summary:**")
            st.write(row['summary'])
else:
    st.info("No summaries found for this filter.")