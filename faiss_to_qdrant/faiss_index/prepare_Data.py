import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import csv

# Configuration
DATA_PATH = '../data'
TSV_FILE = f'{DATA_PATH}/collection.tsv'
SAMPLE_SIZE = 100000
MODEL_ID = 'all-MiniLM-L6-v2'

def prepare_data():
    print(f"1. Loading Model '{MODEL_ID}'...")
    model = SentenceTransformer(MODEL_ID)
    
    print(f"2. Reading first {SAMPLE_SIZE} lines from {TSV_FILE}...")
    
    ids = []
    passages = []
    
    # Efficiently read line-by-line without loading entire 8GB file to RAM
    try:
        with open(TSV_FILE, 'r', encoding='utf8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= SAMPLE_SIZE:
                    break
                
                # MS MARCO format is: [pid, text]
                # Sometimes rows are malformed, so we add a safety check
                if len(row) >= 2:
                    ids.append(int(row[0]))
                    passages.append(row[1])
                    
    except FileNotFoundError:
        print(f"Error: Could not find {TSV_FILE}")
        print("Please ensure 'collection.tsv' is in the 'faiss_to_qdrant/data/' folder.")
        return

    print(f"   Loaded {len(passages)} passages.")
    
    # Save text metadata (for Qdrant payload)
    print("3. Saving metadata to CSV...")
    df = pd.DataFrame({'id': ids, 'text': passages})
    df.to_csv(f'{DATA_PATH}/passages.csv', index=False)
    
    # Generate Embeddings
    print("4. Encoding Embeddings (this may take a moment)...")
    embeddings = model.encode(passages, show_progress_bar=True)
    
    # Save binary files (for FAISS and Qdrant)
    print("5. Saving numpy arrays...")
    np.save(f'{DATA_PATH}/embeddings.npy', embeddings)
    np.save(f'{DATA_PATH}/ids.npy', np.array(ids))
    
    print(f"Success! Saved {embeddings.shape} embeddings to {DATA_PATH}")

if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    prepare_data()