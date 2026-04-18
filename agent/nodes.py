from __future__ import annotations
import json
import os, re
from typing import Any, Dict, List, Optional, TypedDict
from google import genai
from retrieval.rerank_search import hybrid_search_with_rerank
MODEL_NAME = "gemini-2.5-flash"

class IncidentState(TypedDict, total=False):
    incident: str
    analysis: Dict[str, Any]
    # ReAct Fields
    thought: str
    action: str
    action_input: str
    observation: str
    iterations: int
    retrieved_docs: List[Dict[str,Any]]
    final_answer: Dict[str,Any]

_genai_client: Optional[genai.Client] = None

def get_client() -> genai.Client:
    global _genai_client
    if _genai_client:
        return _genai_client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEmini APi KEy")
    _genai_client = genai.Client(api_key=api_key)
    return _genai_client

def _generate(prompt: str) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON Found")
    return json.loads(match.group(0))

# Node1: Analyze
def analyze_incident_node(state: IncidentState) -> Dict[str, Any]:
    incident = state["incident"]
    return {
        "analysis": {
            "summary": incident
        },
        "iterations": 0
    }

# ReACT Agent Node
def react_agent_node(state: IncidentState) -> Dict[str, Any]:
    incident = state["incident"]
    observation = state.get("observation", "")
    iterations = state.get("iterations", 0)
    if iterations > 3:
        return {
            "action": "finish"
        }
    prompt = f"""
    You are an incident triage agent using ReAct.
    You can: 
    1. Search : retieve incident knowledge
    2. finish: return final triage
    Return JSON:
    {{
        "thought": "...",
        "action" : "search or finish",
        "action_input": "query or final answer"
    }}    
    Incident:
    {incident}

    Previous observation:
    {observation}
    """
    output = _generate(prompt)
    parsed = _extract_json(output)
    return {
        "thought": parsed.get("thought"),
        "action": parsed.get("action"),
        "action_input": parsed.get("action_input"),
        "iterations": iterations+1
    }

# Tool Node - search
def tool_node(state: IncidentState) -> Dict[str, Any]:
    action = state.get("action")
    if action == "search":
        query = state.get("action_input", "")
        results = hybrid_search_with_rerank(
            query=query,
            final_limit=5
        )
        return {
            "retrieved_docs": results,
            "observation": json.dumps(results)[:1000]
        }
    return {}

# Final Answer node
def final_node(state: IncidentState) -> Dict[str, Any]:
    incident = state["incident"]
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(
        [doc.get("text", "") for doc in docs]
    )
    prompt = f"""
    Generate the final triage plan.
    Return JSON: 
    {{
        "incident_type" : "",
        "hypothesis": [],
        "next_steps": [],
        "evidence" : []
    }}
    Incident:
    {incident}

    Context:
    {context}
"""
    output = _generate(prompt)
    parsed = _extract_json(output)
    return {
        "final_answer": parsed
    }



























