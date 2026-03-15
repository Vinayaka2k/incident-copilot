import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path: Path) -> List[Dict]:
    """
    Load chunked documents from json
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of text
    """
    if not texts:
        return []
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.tolist()

def attach_embeddings(chunks: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
    """
    Attach embedding vectors to each chunk record
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match")
    embedded_chunks = []
    for chunk, embedding in zip(chunk, embeddings):
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
    model = SentenceTransformer("")
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts, model=model, batch_size=32)
    embedded_chunks = attach_embeddings(chunks, embeddings)

    save_embedded_chunks(output_path, embedded_chunks)
    print(f"Loaded {len(chunks)} chunks")
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Saved embedded chunks to {output_path}")

 











