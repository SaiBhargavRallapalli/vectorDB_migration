from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize App & Resources
app = FastAPI()
client = QdrantClient("http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION_NAME = "ms_marco_passages"

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3

@app.post("/search")
def search(req: QueryRequest):
    # 1. Vectorize
    vector = model.encode(req.text).tolist()
    
    # 2. Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=req.top_k
    ).points
    
    # 3. Format Response
    return {
        "query": req.text,
        "results": [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload['text']
            } for hit in results
        ]
    }

# Run with: uvicorn faiss_to_qdrant.api:app --reload