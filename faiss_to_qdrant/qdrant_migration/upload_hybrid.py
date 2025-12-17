import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding # New Library

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_hybrid"
DATA_PATH = '../data'
BATCH_SIZE = 100

def upload_hybrid():
    client = QdrantClient(url=QDRANT_URL)
    
    print("1. Loading Models...")
    # Dense Model (Context)
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Sparse Model (Keywords - uses SPLADE by default)
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    print("2. Loading Data...")
    df_meta = pd.read_csv(f'{DATA_PATH}/passages.csv')
    # Limit to 1000 for this demo to save time
    df_meta = df_meta.head(1000) 
    
    total = len(df_meta)
    print(f"Starting Hybrid Upload for {total} documents...")
    
    points_batch = []
    
    for i, row in df_meta.iterrows():
        text = row['text']
        
        # A. Generate Dense Vector
        dense_vec = dense_model.encode(text).tolist()
        
        # B. Generate Sparse Vector (Returns a generator, we take the first item)
        # fastembed returns values and indices (non-zero elements)
        sparse_output = list(sparse_model.embed(text))[0]
        
        # Construct Payload
        payload = {
            "passage_id": int(row['id']),
            "text": text,
            "text_length": len(text)
        }
        
        # Add to batch with Named Vectors
        points_batch.append(PointStruct(
            id=int(row['id']),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_output.indices.tolist(),
                    values=sparse_output.values.tolist()
                )
            },
            payload=payload
        ))
        
        if len(points_batch) >= BATCH_SIZE or i == total - 1:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            points_batch = []
            print(f"  Processed {i+1}/{total}...")

    print("Hybrid Upload Complete.")

if __name__ == "__main__":
    upload_hybrid()