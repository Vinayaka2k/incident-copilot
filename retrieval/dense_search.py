from pathlib import Path
from typing import List, Dict

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json
COLLECTION_NAME = "incident_copilot-new"  # Match your index name
OPENSEARCH_ENDPOINT = "https://dww2xv3ocmzsmn17zb66.us-east-1.aoss.amazonaws.com"
REGION = "us-east-1"
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v2:0"

def get_opensearch_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",
        session_token=credentials.token
    )
    host = OPENSEARCH_ENDPOINT.replace("https://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )

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

def dense_search(client: OpenSearch, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    query_body = {
        "size": limit,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": limit
                }
            }
        }
    }

    response = client.search(index=collection_name, body=query_body)

    formatted_results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        formatted_results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "text": source.get("text"),
            "source": source.get("source"),
            "filename": source.get("filename"),
            "doc_type": source.get("doc_type"),
            "chunk_index": source.get("chunk_index"),
            "global_chunk_id": source.get("global_chunk_id")
        })
    return formatted_results

if __name__ == "__main__":
    bedrock_client = get_bedrock_client()
    client = get_opensearch_client()
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