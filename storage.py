# storage.py
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import os  # For directory creation

DB_PATH = "/app/data/papers.db"
def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Ensure directory exists
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS summaries
                    (id INTEGER PRIMARY KEY, date TEXT, paper_json TEXT, summary TEXT)''')
    conn.close()


# DB connection
def get_summaries(weeks_back=1, keyword=None):
    conn = sqlite3.connect("/app/data/papers.db")
    start_date = (datetime.now() - timedelta(weeks=weeks_back)).isoformat()
    query = "SELECT date, paper_json, summary FROM summaries WHERE (date >= ? OR date IS NULL)"
    params = [start_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if keyword:
        lower_keyword = keyword.lower()
        # Parse paper_json to get title/abstract
        df['title'] = df['paper_json'].apply(lambda x: json.loads(x).get('title', '').lower())
        df['abstract'] = df['paper_json'].apply(lambda x: json.loads(x).get('abstract', '').lower())
        df['summary_lower'] = df['summary'].str.lower()
        
        mask = (
            df['title'].str.contains(lower_keyword) |
            df['abstract'].str.contains(lower_keyword) |
            df['summary_lower'].str.contains(lower_keyword)
        )
        df = df[mask]
    
    return df.drop(columns=['title', 'abstract', 'summary_lower'] if 'title' in df.columns else [], errors='ignore')

def store_summaries(papers, summaries, db_path=DB_PATH):
    init_db(db_path)  # Ensure table exists
    conn = sqlite3.connect(db_path)
    for paper, summary in zip(papers, summaries):
        # Handle date for DB insert
        date_value = paper.get('published')
        if isinstance(date_value, datetime):
            date_str = date_value.isoformat()
        elif isinstance(date_value, str):
            date_str = date_value
        else:
            date_str = None
        
        # Prepare paper copy for JSON
        paper_copy = paper.copy()
        if 'published' in paper_copy and isinstance(paper_copy['published'], datetime):
            paper_copy['published'] = paper_copy['published'].isoformat()
        
        conn.execute("INSERT INTO summaries (date, paper_json, summary) VALUES (?, ?, ?)",
                     (date_str, json.dumps(paper_copy), summary))
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM summaries")
    conn.commit()
    conn.close()
