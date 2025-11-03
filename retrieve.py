# retrieve.py
import sqlite3
from datetime import datetime, timedelta
import shared_llm
'''
def retrieve_weekly(db_path="/app/data/papers.db", weeks_back=1):
    conn = sqlite3.connect(db_path)
    start_date = (datetime.now() - timedelta(weeks=weeks_back)).isoformat()
    cursor = conn.execute("SELECT summary FROM summaries WHERE date >= ? OR date IS NULL", (start_date,))
    summaries = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if summaries:
        # Optional: Synthesize into one report via Ollama
        combined = "\n\n".join(summaries)
        synth_prompt = f"Synthesize these paper summaries into a concise weekly update:\n\n{combined}"

        llm = shared_llm.get_hf_llm()
        response = llm.invoke({"instructions": synth_prompt})
        return response['response'].strip()
    return "No updates this week."

# Usage: python retrieve.py
print(retrieve_weekly())
'''
