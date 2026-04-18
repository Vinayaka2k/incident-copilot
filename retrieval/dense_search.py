from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
import boto3
import json

COLLECTION_NAME = "incident_copilot"
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v2:0"

def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant client
    """
    return QdrantClient(host="localhost", port=6333)

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def embed_query(query: str, bedrock_client) -> List[float]:
    payload = {
        "inputText": query.strip(),
        "dimensions": 1024,
        "normalize": True
    }
    response = bedrock_client.invoke_model( 
        modelId=EMBEDDING_MODEL_NAME,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]

def dense_search(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Perform a dense vector search in Qdrant
    """
    response = client.query_points(
        collection_name = collection_name,
        query = query_vector,
        limit = limit,
        with_payload=True
    )
    formatted_results = []
    for hit in response.points:
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
    bedrock_client = get_bedrock_client()
    client = get_qdrant_client()
    query = "payment timeout after deploy"
    query_vector = embed_query(query, bedrock_client)
    results = dense_search(client, COLLECTION_NAME, query_vector, limit=5)
    print(f"Query: {query}")
    print(f"Found {len(results)} results\n")
    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Score: {result['score']}")
        print(f"File: {result['filename']}")
        print(f"Type: {result['doc_type']}")
        print(f"Chunk Id: {result['global_chunk_id']}")
        print(f"Text: {result['text']}")
        print("-"*60)