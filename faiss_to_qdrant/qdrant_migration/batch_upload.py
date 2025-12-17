import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_passages"
DATA_PATH = '../data'
BATCH_SIZE = 500 

def upload_data():
    client = QdrantClient(url=QDRANT_URL)
    
    print("Loading local data...")
    embeddings = np.load(f'{DATA_PATH}/embeddings.npy')
    df_meta = pd.read_csv(f'{DATA_PATH}/passages.csv')

    total = len(df_meta)
    print(f"Starting upload of {total} vectors...")
    
    points_batch = []
    
    for i, row in df_meta.iterrows():
        # Metadata to attach
        payload = {
            "passage_id": int(row['id']),
            "text": row['text'],
            "text_length": len(str(row['text'])),
            "dataset_source": "msmarco_passages" 
        }
        
        points_batch.append(PointStruct(
            id=int(row['id']), 
            vector=embeddings[i].tolist(),
            payload=payload
        ))
        
        # Upload batch
        if len(points_batch) >= BATCH_SIZE or i == total - 1:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            points_batch = [] 
            if i % 1000 == 0:
                print(f"  Processed {i}/{total}...")
                
    print("Upload Complete.")

if __name__ == "__main__":
    upload_data()