from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_hybrid"  #collection name

def create_hybrid_collection():
    client = QdrantClient(url=QDRANT_URL)
    
    print(f"Creating Hybrid Collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        #Dense Vectors (Standard Semantic Search)
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        #Sparse Vectors (Keyword/BM25 Search)
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False, # Keep in RAM for speed
                )
            )
        }
    )
    print("Hybrid Collection Created (Dense + Sparse).")

if __name__ == "__main__":
    create_hybrid_collection()