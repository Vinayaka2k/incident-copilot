import json
from pathlib import Path
from typing import List, Dict
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth
import boto3

INDEX_NAME = "incident_copilot-new"
REGION = "us-east-1"
OPENSEARCH_ENDPOINT = "https://dww2xv3ocmzsmn17zb66.us-east-1.aoss.amazonaws.com"


def load_embedded_chunks(file_path: Path) -> List[Dict]:
    """
    Load embedded chunks from JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch Serverless client with AWS auth
    """
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",  # service name for OpenSearch Serverless
        session_token=credentials.token
    )

    host = OPENSEARCH_ENDPOINT.replace("https://", "")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return client


def create_index(client: OpenSearch, index_name: str, vector_size: int) -> None:
    """
    Create OpenSearch index with knn vector mapping
    """
    if client.indices.exists(index=index_name):
        # Check if we need to recreate
        print(f"Index {index_name} already exists. Deleting and recreating.")
        client.indices.delete(index=index_name)

    body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": vector_size,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "cosinesimil"
                    }
                },
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"},
                "doc_id": {"type": "keyword"},
                "global_chunk_id": {"type": "keyword"}
            }
        }
    }

    client.indices.create(index=index_name, body=body)
    print(f"Created index {index_name} with vector size {vector_size}")


def build_documents(embedded_chunks: List[Dict]) -> List[Dict]:
    """
    Convert embedded chunks into OpenSearch documents (no _id for Serverless)
    """
    documents = []
    for idx, chunk in enumerate(embedded_chunks):
        doc = {
            "_index": INDEX_NAME,
            "_source": {
                "embedding": chunk["embedding"],
                "text": chunk["text"],
                "source": chunk["metadata"].get("source"),
                "filename": chunk["metadata"].get("filename"),
                "doc_type": chunk["metadata"].get("doc_type"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "total_chunks": chunk["metadata"].get("total_chunks"),
                "doc_id": chunk["metadata"].get("doc_id"),
                "global_chunk_id": chunk["metadata"].get("global_chunk_id")
            }
        }
        documents.append(doc)
    return documents


def upload_documents(client: OpenSearch, documents: List[Dict], batch_size: int = 64) -> None:
    """
    Upload documents to OpenSearch in batches
    """
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        success, errors = helpers.bulk(client, batch)
        print(f"Uploaded {min(start + batch_size, len(documents))} / {len(documents)} documents")
        if errors:
            print(f"Errors: {errors}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "processed"
    embedded_chunks_path = processed_dir / "embedded_chunks.json"

    embedded_chunks = load_embedded_chunks(embedded_chunks_path)

    if not embedded_chunks:
        raise ValueError("No embedded chunks found")

    vector_size = len(embedded_chunks[0]["embedding"])

    client = get_opensearch_client()
    create_index(client, INDEX_NAME, vector_size)

    documents = build_documents(embedded_chunks)
    upload_documents(client, documents)

    print(f"Indexed {len(documents)} chunks into OpenSearch index {INDEX_NAME}")