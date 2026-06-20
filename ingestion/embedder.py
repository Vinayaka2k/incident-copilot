import json
import os
from pathlib import Path
from typing import List, Dict
from google import genai
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "gemini-embedding-2"
EMBEDDING_DIM = 3072

_genai_client = None

def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client

def load_chunks(chunks_path: Path) -> List[Dict]:
    """
    Load chunked documents from json
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Gemini gemini-embedding-2.
    """
    if not texts:
        return []
    client = get_genai_client()
    embeddings = []
    
    for text in tqdm(texts, desc="Embedding chunks with Gemini"):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=text.strip()
            )
            embeddings.append(response.embeddings[0].values)
        except Exception as e:
            print(f"Failed to embed text: {text[:50]}... Error: {e}")
            raise e
            
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
    chunks_path = processed_dir / "chunks.json"
    output_path = processed_dir / "embedded_chunks.json"

    chunks = load_chunks(chunks_path)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts)
    embedded_chunks = attach_embeddings(chunks, embeddings)

    save_embedded_chunks(output_path, embedded_chunks)
    print(f"Loaded {len(chunks)} chunks")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Saved embedded chunks to {output_path}")
