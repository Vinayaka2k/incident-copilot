from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from hybrid_search import hybrid_search
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker() -> CrossEncoder:
    """
    Load and return a cross encoder rereanker model
    """
    return CrossEncoder(RERANK_MODEL_NAME)

def rerank_results(query: str, candidates: List[Dict[str, Any]],
                   reranker: CrossEncoder) -> List[Dict[str, Any]]:
    """
    Rerank canddiate results using a cross encoder
    Each candidiate is scored using a pair (query, candidate_text)
    Higher score means more relevant
    """
    if not query or not query.strip():
        raise ValueError("Query cant be empty")
    if not candidates:
        return []
    pairs = []
    valid_candidates = []
    for candidate in candidates:
        text = candidate.get("text")
        if text and text.strip():
            pairs.append((query.strip(), text.strip()))
            valid_candidates.append(candidate)

    if not pairs:
        return []
    
    scores = reranker.predict(pairs)
    reranked_results = []
    for candidate, score in zip(valid_candidates, scores):
        result = dict(candidate)
        result["rerank_score"] = float(score)
        reranked_results.append(result)

    reranked_results.sort(key = lambda x:x["rerank_score"], reverse=True)
    return reranked_results

def hybrid_search_with_rerank(query: str, dense_limit: int = 10, keyword_limit: int = 10,
                              hybrid_limit: int = 10, final_limit: int = 5) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval first, then rerank retreieved candidaites.
    Stage 1: hybrid retrieval for reacll
    Stage 2: cross encoder reranking for prevision
    """
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be rmpty")
    candidates = hybrid_search(query=query, dense_limit=dense_limit,
                               keyword_limit=keyword_limit, final_limit=final_limit, rrf_k=60)
    reranker = get_reranker()
    reranked = rerank_results(query, candidates, reranker)
    return reranked[:final_limit]

if __name__ == "__main__":
    query = "payment timeout after deploy"
    results = hybrid_search_with_rerank(query=query, dense_limit=10, keyword_limit=10, hybrid_limit=10, final_limit=5)
    print(f"QUery: {query}")
    print(f"Found {len(results)} reranked results\n")
    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Rerank score: {result.get('rerank_score')}")
        print(f"Hybrid score: {result.get('hybrid_score')}")
        print(f"Sources: {result.get('retrieval_sources')}")
        print(f"File: {result.get('filename')}")
        print(f"Type: {result.get('doc_type')}")
        print(f"Chunk id: {result.get('global_chunk_id')}")
        print(f"Text: {result.get('text')}")
        print("-"*70)











