from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_passages"

def create_collection():
    client = QdrantClient(url=QDRANT_URL)
    
    print(f"Creating collection '{COLLECTION_NAME}'...")
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,             # Dimension (MiniLM) - we should follow the existing dimension used in FAISS
            distance=Distance.COSINE
        ),
        hnsw_config=HnswConfigDiff(
            m=16,                 # Links per node (default is 16)
            ef_construct=100      # Search depth during build (default is 100)
        )
    )
    
    print(f"✅ Collection '{COLLECTION_NAME}' created with HNSW config.")

if __name__ == "__main__":
    create_collection()