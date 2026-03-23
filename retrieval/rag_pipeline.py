from typing import List, Dict, Any
from google import genai
from rerank_search import hybrid_search_with_rerank

MODEL_NAME = "gemini-2.5-flash"

def get_client():
    """
    Initialize Gemini client.
    Assumes GEMINI_API_KEY has been set
    """
    return genai.Client(api_key="AIzaSyCThrHM0CA-7ONbuso5rQoi8SiXugXXEfE")

def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Combine retrieved chunks into context string
    """
    parts = []
    for i, result in enumerate(results, start=1):
        text = result.get("text", "")
        filename = result.get("filename", "unknown")
        parts.append(f"[Source {i} - {filename}] \n {text}")
    return "\n\n".join(parts)

def build_prompt(query: str, context: str) -> str:
    """
    Build RAG prompt
    """
    return f"""
    You are an incident analysis assistant
    Use only the provided context to answer the question
    If the answer is not in the context, say you don't know

    Context:
    {context}

    Question:
    {query}

    Answer:    
"""

def generate_answer(query: str, context: str, client) -> str:
    """
    Call gemini model
    """
    prompt = build_prompt(query, context)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

def rag_pipeline(query: str, final_k: int = 5) -> Dict[str, Any]:
    """
    Full pipeline:
    retrieval -> rerank -> context -> LLM
    """
    query = query.strip()
    if not query:
        raise ValueError("query cant be empty")
    results = hybrid_search_with_rerank(
        query=query,
        dense_limit=10,
        keyword_limit=10,
        hybrid_limit=10,
        final_limit=final_k
    )
    context = build_context(results)
    client = get_client()
    answer = generate_answer(query, context, client)
    return {
        "query": query,
        "answer": answer,
        "sources": results
    }

if __name__ == "__main__":
    query = "Why did payment timeouts happen after deployment?"
    output = rag_pipeline(query)
    print(f"Query: {output['query']} \n")
    print(f"Answer: \n")
    print(output['answer'])
    print("\n Sources: \n")
    for i, src in enumerate(output["sources"], start=1):
        print(f"{i}. {src.get('filename')} | Score: {src.get('rerank_score')}")