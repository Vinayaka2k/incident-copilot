import json
import boto3
from typing import List, Dict, Any
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

INDEX_NAME = "incident_copilot-new"
OPENSEARCH_ENDPOINT = "https://dww2xv3ocmzsmn17zb66.us-east-1.aoss.amazonaws.com"
REGION = "us-east-1"


def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch Serverless client
    """
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


def keyword_search(query: str, client: OpenSearch = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    BM25 keyword search using OpenSearch's built-in text matching
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    if client is None:
        client = get_opensearch_client()

    query_body = {
        "size": limit,
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "operator": "or"
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query_body)

    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "text": source.get("text"),
            "source": source.get("source"),
            "filename": source.get("filename"),
            "doc_type": source.get("doc_type"),
            "chunk_index": source.get("chunk_index"),
            "global_chunk_id": source.get("global_chunk_id")
        })

    return results


if __name__ == "__main__":
    client = get_opensearch_client()
    query = "payment timeout after deploy"
    results = keyword_search(query, client=client, limit=5)

    print(f"Query: {query}")
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Score: {result['score']}")
        print(f"File:  {result['filename']}")
        print(f"Type: {result['doc_type']}")
        print(f"Chunk id: {result['global_chunk_id']}")
        print(f"Text: {result['text']}")
        print("-" * 60)