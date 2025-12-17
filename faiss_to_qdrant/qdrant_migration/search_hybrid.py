from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Fusion, FusionQuery, SparseVector
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_hybrid"

def hybrid_search_demo():
    client = QdrantClient(url=QDRANT_URL)
    
    # Load Models
    print("Loading models...")
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # QUERY: A mix of concept and specific keyword
    query_text = "What is the capital of France?"
    print(f"--- Hybrid Query: '{query_text}' ---")
    
    # 1. Generate Vectors for Query
    dense_q = dense_model.encode(query_text).tolist()
    # fastembed returns a generator, so we list() it and take the first item
    sparse_q = list(sparse_model.embed(query_text))[0]
    
    # 2. Execute Hybrid Search (Fusion)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        
        # Run TWO searches in parallel (Prefetch)
        prefetch=[
            # A. Dense Search (Best for meaning)
            Prefetch(
                query=dense_q,
                using="dense",
                limit=10
            ),
            # B. Sparse Search (Best for exact keywords)
            Prefetch(
                query=SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist()
                ),
                using="sparse",
                limit=10
            ),
        ],
        
        # --- THE FIX IS HERE ---
        # Changed 'method' to 'fusion'
        query=FusionQuery(fusion=Fusion.RRF),
        limit=3
    ).points
    
    # 3. Print Results
    for hit in results:
        print(f"\n[Score: {hit.score:.3f}] ID: {hit.id}")
        print(f"Text: {hit.payload['text']}")

if __name__ == "__main__":
    hybrid_search_demo()