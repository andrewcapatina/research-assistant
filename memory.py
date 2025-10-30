from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # Dimension for model

def add_to_memory(summary):
    embedding = embedder.encode(summary)
    index.add(np.array([embedding]))

def query_memory(query, k=3):
    embedding = embedder.encode(query)
    _, indices = index.search(np.array([embedding]), k)
    # Retrieve from DB using indices (map to IDs)
    return indices  # Stub; expand to fetch summaries