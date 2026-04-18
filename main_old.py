from typing import List, Dict, Any, Optional
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from retrieval.rerank_search import hybrid_search_with_rerank
MODEL_NAME = "gemini-2.5-flash"
app = FastAPI(
    title = "Incident CoPilot RAG API",
    description="FastAPI service for hybrid retrieval and reranking and gemini asnwer generation",
    version = "1.0.0"
)
genai_client: Optional[genai.Client] = None

class QueryRequest(BaseModel):
    query : str = Field(..., min_length=1, description="USer Query")
    final_k: int = Field(5, ge=1, le=20, description="Number of final reranked chunks")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]

def get_client() -> genai.Client:
    """
    Iniitalize Gemini client from environment variable
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY envionrment variable is not set")
    return genai.Client(api_key=api_key)

def build_context(results: List[Dict[str, Any]]) -> str:
    """
    Combine retreived chunks into a single context string
    """
    parts = []
    for i, result in enumerate(results, start=1):
        text = result.get("text", "")
        filename = result.get("filename", "unknown")
        parts.append(f"[Source {i} - {filename}] \n {text}")
    return "\n\n".join(parts)

def build_prompt(query: str, context: str) -> str:
    """
    Build the RAG prompt
    """
    return f"""
    You are an incident analysis assistnt. Use only the 
    provided context to answer the qquestion. If the answer is not
    in the contetx, say you dont know.

    Context:
    {context}

    Question:
    {query}

    Answer:   
    """.strip()

def generate_answer(query: str, context: str, client: genai.Client) -> str:
    """
    Genertae answer from gemini
    """
    prompt = build_prompt(query, context)
    response = client.models.generate_content(
        model = MODEL_NAME,
        contents = prompt
    )
    if not getattr(response, "text", None):
        raise ValueError("Model returned an empty response")
    return response.text.strip()

def rag_pipeline(query: str, final_k: int = 5) -> Dict[str, Any]:
    """
    Full online inference pipeline
    query -> hybrid retirval -> reranking -> context -> LLM
    """
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    results= hybrid_search_with_rerank(query=query,
                                       dense_limit=10,
                                       keyword_limit=10,
                                       hybrid_limit=10,
                                       final_limit=final_k)
    context = build_context(results)
    client = genai_client if genai_client is not None else get_client()
    answer = generate_answer(query, context, client)
    return {
        "query": query,
        "answer": answer,
        "sources": results
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        result = rag_pipeline(request.query, request.final_k)
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
















