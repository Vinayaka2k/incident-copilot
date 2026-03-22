from typing import List, Dict, Any, Tuple
from dense_search import (
    get_qdrant_client,
    get_embedding_model,
    embed_query,
    dense_search,
    COLLECTION_NAME
)
from keyword_search import keyword_search

def make_result_key(result: Dict[str, Any]) -> Tuple:
    return (
        result.get("id"),
        result.get("global_chunk_id"),
        result.get("filename"),
        result.get("chunk_index"),
        result.get("text")
    )

def reciprocal_rank_fusion(ranked_lists: List[List[Dict[str,Any]]],
                           k: int = 60) -> List[Dict[str,Any]]:
    fused = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            key = make_result_key(result)
            if key not in fused:
                fused[key] = {
                    "rrf_score": 0.0,
                    "result": dict(result),
                    "sources": []
                }
            fused[key]["rrf_score"] += 1.0/(k+rank)
            source_type = result.get("_retrieval_source")
            if source_type and source_type not in fused[key]["sources"]:
                fused[key]["sources"].append(source_type)
    merged_results = []
    for item in fused.values():
        merged = item["result"]
        merged["hybrid_score"] = item["rrf_score"]
        merged["retrieval_sources"] = item["sources"]
        merged_results.append(merged)
    merged_results.sort(key=lambda x:x["hybrid_score"], reverse=True)
    return merged_results

def hybrid_search(query: str, dense_limit: int = 10, keyword_limit: int = 10, 
                  final_limit: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")
    model = get_embedding_model()
    client = get_qdrant_client()
    query_vector = embed_query(query, model)
    dense_results = dense_search(
        client = client,
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=dense_limit
    )
    for result in dense_results:
        result["_retrieval_source"] = "dense"
    keyword_results = keyword_search(query, limit=keyword_limit)
    for result in keyword_results:
        result["_retrieval_source"] = "keyword"
    fused_results = reciprocal_rank_fusion(ranked_lists=[dense_results, keyword_results],
                                           k = rrf_k)
    return fused_results[:final_limit]

if __name__ == "__main__":
    query = "payment timeout after deploy"
    results = hybrid_search(query)
    print(f"Query: {query}")
    print(f"Found {len(results)} hybrid results\n")
    for i, result in enumerate(results, start=1):
        print(f"Result: {i}")
        print(f"Hybrid Score: {result.get('hybrid_score')}")
        print(f"Sources: {result.get('retrieval_sources')}")
        print(f"File: {result.get('filename')}")
        print(f"Type: {result.get('doc_type')}")
        print(f"Chunk Id: {result.get('global_chunk_id')}")
        print(f"Text: {result.get('text')}")
        print("-"*60)