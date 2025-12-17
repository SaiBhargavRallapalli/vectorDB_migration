import faiss
import numpy as np
import os

DATA_PATH = '../data'
INDEX_OUTPUT_PATH = './my_index.faiss'

def build_index():
    print("1. Loading embeddings from disk...")
    # Load the vectors you created in Step 3.1
    if not os.path.exists(f'{DATA_PATH}/embeddings.npy'):
        print(f"Error: {DATA_PATH}/embeddings.npy not found.")
        return

    embeddings = np.load(f'{DATA_PATH}/embeddings.npy')
    d = embeddings.shape[1]  # Dimension (should be 384 for MiniLM)
    
    print(f"2. Building Index (Dimension={d})...")
    
    # We use IndexFlatL2 for exact search (Simple & Accurate for <1M vectors).
    # The guide mentions HNSW/IVF, which are faster for massive datasets (10M+),
    index = faiss.IndexFlatL2(d) 
    index.add(embeddings)
    
    print(f"3. Saving index to {INDEX_OUTPUT_PATH}...")
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"Success! Index contains {index.ntotal} vectors.")

if __name__ == "__main__":
    # Ensure the folder exists
    os.makedirs(os.path.dirname(INDEX_OUTPUT_PATH), exist_ok=True)
    build_index()