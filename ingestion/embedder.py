import json
from pathlib import Path
from typing import List, Dict
import boto3
from tqdm import tqdm
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM=1024

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def load_chunks(chunks_path: Path) -> List[Dict]:
    """
    Load chunked documents from json
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_texts(texts: List[str], bedrock_client) -> List[List[float]]:
    """
    Genreate embeddings for a list of texts using amzn titan embed v2 
    titan processes one text at a time no native batching
    """
    if not texts:
        return []
    embeddings = []
    for text in tqdm(texts, desc="Embedding chunks"):
        payload = {
            "inputText": text.strip(),
            "dimensions": EMBEDDING_DIM,
            "normalize": True,
            # "embedding_types": ["float"]
        }
        response = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_NAME,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        response_body = json.loads(response["body"].read())
        embeddings.append(response_body["embeddingsByType"]["float"])
    return embeddings

def attach_embeddings(chunks: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
    """
    Attach embedding vectors to each chunk record
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            "id": chunk["metadata"]["global_chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding
        })
    return embedded_chunks

def save_embedded_chunks(output_path: Path, embedded_chunks: List[Dict]) -> None:
    """
    Save embedded chunks to JSON
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "processed"
    chunks_path = processed_dir/ "chunks.json"
    output_path = processed_dir / "embedded_chunks.json"

    chunks = load_chunks(chunks_path)
    bedrock_client = get_bedrock_client()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts, bedrock_client=bedrock_client)
    embedded_chunks = attach_embeddings(chunks, embeddings)

    save_embedded_chunks(output_path, embedded_chunks)
    print(f"Loaded {len(chunks)} chunks")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Saved embedded chunks to {output_path}")

 











