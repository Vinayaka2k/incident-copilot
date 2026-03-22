from typing import List, Dict
import json

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#     if overlap >= chunk_size:
#         raise ValueError("Overlap must be smaller than the chunk size")
#     step = chunk_size - overlap
#     for start in range(0, len(text), step):
#         end = min(len(text), start + chunk_size)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         if end == len(text):
#             break
#     return chunks

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks.
    Args:
        text: Full document text
        chunk_size: Max characters per chunk
        overlap: Number of overlapping characters between chunks
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller thank chunk_size")

    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start+chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap
    return chunks

def chunk_documents(documents: List[Dict], chunk_size: int = 600, overlap: int = 100) -> List[Dict]:
    """
    Split loaded documents into chunks and perserve metadata.
    Args:
        documents: Output from loader.py
        chunk_size: Max characters per chunk
        overlap: Overlap between 2 chunks
    Returns:
        List of chunk dictionries with text and metadata
    """
    all_chunks = []
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["doc_id"] = metadata["filename"]
            chunk_metadata["global_chunk_id"] = f"{metadata["filename"]}_chunk_{i}"
            all_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
    return all_chunks


if __name__ == "__main__":
    from loader import load_markdown_files
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    docs = load_markdown_files(data_dir)
    # print(docs)
    chunks = chunk_documents(docs, chunk_size=600, overlap=100)
    output_file = output_dir / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved chunks to {output_file}")







