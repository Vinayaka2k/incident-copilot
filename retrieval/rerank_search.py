from typing import List, Dict, Any
from hybrid_search import hybrid_search
RERANK_MODEL_NAME = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
import boto3

def get_reranker():
    return boto3.client("bedrock-agent-runtime", region_name="us-east-1")

def rerank_results(query: str, candidates: List[Dict[str,Any]], reranker) -> List[Dict[str,Any]]:
    """
    Rerank candaites using amanzn bedrock rerank API
    """
    if not query or not query.strip():
        raise ValueError("Query cant be empty")
    if not candidates:
        return []
    valid_candidates = []
    sources = []
    for candidate in candidates:
        text = candidate.get("text")
        if text and text.strip():
            valid_candidates.append(candidate)
            sources.append({
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {
                        "text": text.strip()
                    }
                }
            })
    if not sources:
        return []
    response = reranker.rerank(
        queries=[
            {
                "type": "TEXT", "textQuery": {
                    "text": query.strip()
                }
            }
        ],
        sources=sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration" : {
                "modelConfiguration": {
                    "modelArn": RERANK_MODEL_NAME
                }, "numberOfResults": len(valid_candidates)
            }
        }
    )
    reranked_results = []
    for result in response["results"]:
        idx=result["index"]
        candidate=dict(valid_candidates[idx])
        candidate["rerank_score"]=float(result["relevanceScore"])
        reranked_results.append(candidate)
    reranked_results.sort(key=lambda x:x["rerank_score"], reverse=True)
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











