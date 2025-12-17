import time
import faiss
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration
FAISS_INDEX_PATH = './faiss_index/my_index.faiss'
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_passages"
MODEL_NAME = 'all-MiniLM-L6-v2'

# 10 Representative Queries
QUERIES = [
    "What is a GPU?",
    "Who is the president of France?",
    "How to bake a cake?",
    "Symptoms of the flu",
    "Python programming language",
    "Best places to visit in Italy",
    "Define quantum mechanics",
    "History of the Roman Empire",
    "What is machine learning?",
    "Healthy breakfast ideas"
]

def run_comparison():
    print("--- 1. Loading Resources ---")
    
    # Load Model
    model = SentenceTransformer(MODEL_NAME)
    
    # Load FAISS (The "Old Way")
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Connect to Qdrant (The "New Way")
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    print(f"\n--- 2. Running Race ({len(QUERIES)} queries) ---")
    print(f"{'Query':<30} | {'FAISS (ms)':<10} | {'Qdrant (ms)':<10}")
    print("-" * 60)

    faiss_times = []
    qdrant_times = []

    for query_text in QUERIES:
        # Encode once
        query_vector = model.encode(query_text).tolist()
        
        # --- MEASURE FAISS ---
        start_f = time.perf_counter()
        # FAISS expects a numpy array of shape (1, d)
        faiss_input = np.array([query_vector], dtype='float32')
        _, _ = faiss_index.search(faiss_input, k=3)
        end_f = time.perf_counter()
        faiss_ms = (end_f - start_f) * 1000
        faiss_times.append(faiss_ms)

        # --- MEASURE QDRANT ---
        start_q = time.perf_counter()
        _ = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        end_q = time.perf_counter()
        qdrant_ms = (end_q - start_q) * 1000
        qdrant_times.append(qdrant_ms)

        print(f"{query_text[:30]:<30} | {faiss_ms:>10.2f} | {qdrant_ms:>10.2f}")

    print("-" * 60)
    print(f"{'AVERAGE':<30} | {np.mean(faiss_times):>10.2f} | {np.mean(qdrant_times):>10.2f}")

if __name__ == "__main__":
    run_comparison()