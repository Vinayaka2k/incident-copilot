import json
from pathlib import Path
from typing import List, Dict
from opensearchpy import OpenSearch, RequestsHttpConnection,  helpers
from requests_aws4auth import AWS4Auth
import boto3

INDEX_NAME = "incident-copilot-new"
REGION="us-east-1"

OPENSEARCH_ENDPOINT = ""

def load_embedded_chunks(file_path: Path) -> List[Dict]:
    """
    Load embedded chunks from JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_opensearch_client() -> OpenSearch:
    """
    Create and return an opensarch serverless client with aws auth
    """
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "aoss",
        session_token=credentials.token
    )
    host = OPENSEARCH_ENDPOINT.replace("http://", "")
    client = OpenSearch(
        hosts=[
            {
                "host": host, "port": 443
            }
        ],
        http_auth=auth,
        use_ssl=True,
        verify_cert=True,
        connection_class=RequestsHttpConnection
    )
    return client

def create_index(client: OpenSearch, index_name: str, vector_size: int) -> None:
    """
    create opensearch index with knn vector mapping
    """
    if client.indices.exists(index=index_name):
        print(f"index  {index_name} already exists. deleteing and creating again")
        client.indices.delete(index=index_name)
    body = {
        "settings": {
            "index": {
                "knn": True                
            }
        }
    }, "mappings": {
            "properties" : {
                "embedding": {
                    "type":  "knn_vector",
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
                "total_chunks": {"type": "inetgeer"},
                "doc_id": {"type": "keyword"},
                "global_chunk_id": {"type": "keyword"}
            }
            client.indcices.create(index=INDEX_NAME, body=body)
            print(f"Created index {INDEX_NAME} with vecor size {vector_size}")
    }























