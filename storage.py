# storage.py
import sqlite3
import json
from datetime import datetime
import os  # For directory creation

def init_db(db_path="/app/data/papers.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)  # Ensure directory exists
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS summaries
                    (id INTEGER PRIMARY KEY, date TEXT, paper_json TEXT, summary TEXT)''')
    conn.close()

def store_summaries(papers, summaries, db_path="/app/data/papers.db"):
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