import json
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25OKapi
CHUNKS_PATH = Path("processed/chunks.json")

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25
    """
    if not text:
        return []
    return [token for token in re.split(r"\W+", text.lower()) if token]

def load_chunks(chunks_path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """
    Load chunked documents from processed/chunks.json
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must containa list of chunk records")
    return chunks

class BM25KeywordSearcher:
    """
    BM25 based keyword search over local chunked documents
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            raise ValueError("No chunks provided to BM25 searcher")
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.bm25 = BM25OKapi(self.tokenized_corpus)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Return bm25 searhc results in a format consistent with dense_search.py
        """
        query  = query.strip()
        if not query:
            raise ValueError("Query cannot be emty")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            results.append({
                "id": chunk.get("id", idx),
                "score": float(score),
                "text": chunk.get("text"),
               "source": chunk.get("source") or chunk.get("metadata", {}).get("source"),
                "filename": chunk.get("filename") or chunk.get("metadata", {}).get("filename"),
                "doc_type": chunk.get("doc_type") or chunk.get("metadata", {}).get("doc_type"),
                "chunk_index": chunk.get("chunk_index") if chunk.get("chunk_index") is not None else chunk.get("metadata", {}).get("chunk_index"),
                "global_chunk_id": chunk.get("global_chunk_id") if chunk.get("global_chunk_id") is not None else chunk.get("metadata", {}).get("global_chunk_id")
            })
        results.sort(key = lambda x: x["score"], reverse=True)
        return [result for result in results if result["score"] > 0][:limit]

def keyword_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    One shot bm25 keyword search
    """
    chunks = load_chunks()
    searcher = BM25KeywordSearcher(chunks)
    return searcher.search(query, limit=limit)

if __name__ == "__main__":
    query = "payment timeout after deploy"
    results = keyword_search(query, limit=5)
    print(f"Query: {query}")
    print(f"Found {len(results)} results \n")

    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Score: {result['score']}")
        print(f"File:  {result['filename']}")
        print(f"Type: {result['doc_type']}")
        print(f"Chunk id: {result['global_chunk_id']}")
        print(f"text: {result['text']}")
        print("-"*60)