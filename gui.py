import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import ollama  # For optional synthesis
from io import StringIO
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import planner_agent

# DB connection
def get_summaries(weeks_back=1, keyword=None):
    conn = sqlite3.connect("/app/data/papers.db")
    start_date = (datetime.now() - timedelta(weeks=weeks_back)).isoformat()
    query = "SELECT date, paper_json, summary FROM summaries WHERE (date >= ? OR date IS NULL)"
    params = [start_date]
    if keyword:
        query += " AND (summary LIKE ? OR paper_json LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Build FAISS index for semantic search (from all summaries in DB)
@st.cache_resource  # Cache for performance
def build_faiss_index():
    conn = sqlite3.connect("/app/data/papers.db")
    df_all = pd.read_sql_query("SELECT id, summary FROM summaries", conn)
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

df = get_summaries(weeks_back=weeks, keyword=keyword)

if not df.empty:
    # Optional: Synthesize if checkbox selected
    if st.checkbox("Synthesize into report (via Ollama)"):
        summaries = df['summary'].tolist()
        combined = "\n\n".join(summaries)
        synth_prompt = f"Compile these distinct paper summaries into a unique, non-repetitive report. Structure as: 1. Overview of key themes. 2. Bullet points of major innovations per paper (no duplicates). 3. Implications for AI/comp arch. Be concise, avoid filler:\n\n{combined[:3000]}"
        with st.spinner("Synthesizing..."):
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=synth_prompt,
                options={"temperature": 0.4, "top_p": 0.8, "repeat_penalty": 1.3}
            )
        st.markdown("### Synthesized Report")
        st.write(response['response'].strip())

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
    with st.spinner("Planning and executing..."):
        result = planner_agent.run_planner(goal_input)
    st.markdown("### Planner Results")
    
    # Display structured output
    with st.expander("Generated Plan"):
        st.write(result.get("plan", "No plan generated."))
    
    summaries = result.get("summaries", [])
    if summaries:
        st.markdown("### Summaries")
        for i, summary in enumerate(summaries, 1):
            st.write(f"**Summary {i}:** {summary}")
    
    with st.expander("Evaluation"):
        st.write(result.get("evaluation", "No evaluation generated."))

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