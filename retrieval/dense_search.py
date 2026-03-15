from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "incident_copilot"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client
    """
    return QdrantClient(host="localhost", port=6334)

def get_embedding_model() -> SentenceTransformer:
    """
    Load and return the embedding model
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_query(query: str, model: SentenceTransformer) -> List[float]:
    """
    Convert a user query into a dense embedding vector
    """
    query_text = f"query: {query.strip()}"
    embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.tolist()

def dense_search(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Perform a dense vector search in Qdrant
    """
    results = client.search(
        collection_name = collection_name,
        query_vector = query_vector,
        limit = limit
    )
    formatted_results = []
    for hit in results:
        formatted_results.append({
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text"),
            "source": hit.payload.get("source"),
            "filename": hit.payload.get("filename"),
            "doc_type": hit.payload.get("doc_type"),
            "chunk_index": hit.payload.get("chunk_index"),
            "global_chunk_id": hit.payload.get("global_chunk_id")
        })
    return formatted_results

if __name__ == "__main__":
    model = get_embedding_model()
    client = get_qdrant_client()
    query = "payment timeout after deploy"
    query_vector = embed_query(query, model)
    results = dense_search(client, COLLECTION_NAME, query_vector, limit=5)
    print(f"Query: {query}")
    print(f"Found {len(results)} results\n")
    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Score: {result["score"]}")
        print(f"File: {result["filename"]}")
        print(f"Type: {result["doc_type"]}")
        print(f"Chunk Id: {result["global_chunk_id"]}")
        print(f"Text: {result["text"]}")
        print("-"*60)