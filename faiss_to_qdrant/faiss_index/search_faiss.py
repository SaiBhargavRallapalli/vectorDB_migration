import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = 'faiss_to_qdrant/faiss_index/my_index.faiss'
DATA_PATH = '../data'
MODEL_NAME = 'all-MiniLM-L6-v2'

def search_faiss():
    print("1. Loading Index and Metadata...")
    index = faiss.read_index(INDEX_PATH)
    
    # LIMITATION: We must manually load the CSV to get text back.
    # FAISS only stores vectors, not the text itself.
    df = pd.read_csv(f'{DATA_PATH}/passages.csv')
    
    model = SentenceTransformer(MODEL_NAME)
    
    # The Query
    query_text = "What is the capital of France?"
    print(f"\nQuery: '{query_text}'")
    
    # Encode and Search
    query_vector = model.encode([query_text])
    D, I = index.search(query_vector, k=3) # Search for top 3
    
    print("\n--- Results ---")
    for rank, idx in enumerate(I[0]):
        # LIMITATION: If we wanted to filter by "text_length > 50", 
        # we would have to fetch ALL results first, then filter in Python. 
        # FAISS cannot filter during search.
        
        text = df.iloc[idx]['text'] # Manual lookup
        score = D[0][rank]
        print(f"[{rank+1}] ID: {idx} | Score: {score:.4f}")
        print(f"     Text: {text[:100]}...")

if __name__ == "__main__":
    search_faiss()