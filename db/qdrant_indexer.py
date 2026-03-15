import json
from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
COLLECTION_NAME = "incident_copilot"
def load_embedded_chunks(file_path: Path) -> List[Dict]:
    """
    Load embedded chunks from JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Create Qdrant collection if it does not already exist
    """
    existing_collections = client.get_collections().collections
    existing_names = {collection.name for collection in existing_collections}
    if collection_name in existing_names:
        print(f"Collection {collection_name} already exists")
        return
    client.create_collection(
        collection_name = collection_name,
        vectors_config = VectorParams(
            size = vector_size,
            distance = Distance.COSINE
        )
    )
    print(f"Created collection {collection_name}")

def build_points(embedded_chunks: List[Dict]) -> List[PointStruct]:
    """
    Convert embedded chunks into Qdrant PointStruct objects
    """
    points = []
    for idx, chunk in enumerate(embedded_chunks):
        payload = {
            "text": chunk["text"],
            "source": chunk["metadata"].get("source"),
            "filename": chunk["metadata"].get("filename"),
            "doc_type": chunk["metadata"].get("doc_type"),
            "chunk_index": chunk["metadata"].get("chunk_index"),
            "total_chunks": chunk["metadata"].get("total_chunks"),
            "doc_id": chunk["metadata"].get("doc_id"),
            "global_chunk_id": chunk["metadata"].get("global_chunk_id")
        }
        points.append(
            PointStruct(
                id = idx,
                vector = chunk["embedding"],
                payload = payload
            )
        )
    return points

def upload_points(client: QdrantClient, collection_name: str, points: List[PointStruct], batch_size: int = 64) -> None:
    """
    Upload points to qdrant in bacthes
    """
    for start in range(0, len(points), batch_size):
        batch = points[start: start+batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Uploaded {start+len(batch)} / {len(points)} points")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "processed"
    embedded_chunks_path = processed_dir / "embedded_chunks.json"
    embedded_chunks = load_embedded_chunks(embedded_chunks_path)
    if not embedded_chunks:
        raise ValueError("No embedded chunks found")
    vector_size = len(embedded_chunks[0]["embedding"])
    client = QdrantClient(host="localhost", port=6334)
    create_collection(client, COLLECTION_NAME, vector_size)
    points = build_points(embedded_chunks)
    upload_points(client, COLLECTION_NAME, points)
    print(f"Indexed {len(points)} chunks into Qdrant collection {COLLECTION_NAME}")






















