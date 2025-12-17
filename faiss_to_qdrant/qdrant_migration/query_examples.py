from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ms_marco_passages"
MODEL_NAME = 'all-MiniLM-L6-v2'

def validate_migration():
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Verify total count
    count_result = client.count(COLLECTION_NAME)
    print(f"Total Vectors in Qdrant: {count_result.count}")

    # 2. Query example
    query_text = "What is the capital of INDIA?"
    print(f"\n--- Query: '{query_text}' ---")
    query_vector = model.encode(query_text).tolist()
    
    # 3. Filter Definition
    print("Applying filters (Length < 200 AND Source == msmarco)...")
    search_filter = Filter(
        must=[
            FieldCondition(
                key="text_length",
                range=Range(lt=1000)
            ),
            FieldCondition(
                key="dataset_source",
                match=MatchValue(value="msmarco_passages")
            )
        ]
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,       
        query_filter=search_filter,
        limit=3
    ).points 
    
    for hit in results:
        print(f"\nID: {hit.id} (Score: {hit.score:.3f})")
        print(f"Text: {hit.payload['text']}")
        print(f"Metadata: {hit.payload}")

if __name__ == "__main__":
    validate_migration()