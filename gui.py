# gui.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import ollama  # For optional synthesis
from io import StringIO

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

# Streamlit app
st.title("Research Paper Summaries Dashboard")

# Filters
weeks = st.slider("Look back weeks:", 1, 4, 1)
keyword = st.text_input("Search keyword (in title/abstract/summary):")

df = get_summaries(weeks_back=weeks, keyword=keyword)

if not df.empty:
    # Optional: Synthesize if checkbox selected
    if st.checkbox("Synthesize into report (via Ollama)"):
        summaries = df['summary'].tolist()
        combined = "\n\n".join(summaries)
        synth_prompt = f"Compile these distinct paper summaries into a unique, non-repetitive report. Structure as: 1. Overview of key themes. 2. Bullet points of major innovations per paper (no duplicates). 3. Implications for AI/comp arch. Be concise, avoid filler:\n\n{combined[:3000]}"  # Stricter truncate
        with st.spinner("Synthesizing..."):
            response = ollama.generate(
                model="llama3.1:8b",
                prompt=synth_prompt,
                options={"temperature": 0.4, "top_p": 0.8, "repeat_penalty": 1.3}  # Tuned to minimize loops
            )
        st.markdown("### Synthesized Report")
        st.write(response['response'].strip())

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