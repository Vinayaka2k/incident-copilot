from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from google import genai
from retrieval.rerank_search import hybrid_search_with_rerank

MODEL_NAME = "gemini-2.5-flash"


class IncidentState(TypedDict, total=False):
    """
    Shared state passed across the LangGraph nodes.
    """
    incident: str
    analysis: Dict[str, Any]
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    incident_type: str
    hypotheses: List[str]
    next_steps: List[str]
    evidence: List[Dict[str, str]]


_genai_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """
    Initialize and cache the Gemini client from GEMINI_API_KEY.
    """
    global _genai_client

    if _genai_client is not None:
        return _genai_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _client() -> genai.Client:
    """
    Return a reusable Gemini client.
    """
    return get_client()


def _generate_text(prompt: str, client: Optional[genai.Client] = None) -> str:
    """
    Helper to call Gemini and return plain text output.
    """
    active_client = client or _client()

    response = active_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Model returned an empty response")

    return text.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from the model output.

    Handles:
    - raw JSON
    - fenced JSON blocks
    - text containing a JSON object
    """
    text = text.strip()

    # Case 1: raw JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Case 2: fenced JSON block
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL,
    )
    if fenced_match:
        candidate = fenced_match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Case 3: loose JSON object inside extra text
    loose_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if loose_match:
        candidate = loose_match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON object from the model output")


def _safe_list_of_strings(value: Any) -> List[str]:
    """
    Normalize a value into a list of non-empty strings.
    """
    if not isinstance(value, list):
        return []

    output: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            output.append(text)

    return output


def _build_context(results: List[Dict[str, Any]]) -> str:
    """
    Combine retrieved chunks into a single grounded context string.
    """
    parts: List[str] = []

    for i, result in enumerate(results, start=1):
        filename = result.get("filename", "unknown")
        chunk_index = result.get("chunk_index", "unknown")
        text = str(result.get("text", "")).strip()

        parts.append(
            f"[Source {i}] filename={filename}, chunk_index={chunk_index}\n{text}"
        )

    return "\n\n".join(parts)


def analyze_incident_node(state: IncidentState) -> Dict[str, Any]:
    """
    Node 1: Lightweight incident analysis with no LLM call.

    Purpose:
    - validate incident input
    - extract simple surface-level signals
    - prepare a minimal analysis object for downstream nodes
    """
    incident = state.get("incident", "").strip()
    if not incident:
        raise ValueError("incident cannot be empty")

    lowered = incident.lower()

    keywords: List[str] = []
    known_terms = [
        "timeouterror",
        "deadline exceeded",
        "crashloopbackoff",
        "redis",
        "kafka",
        "latency",
        "timeout",
        "worker",
        "deploy",
        "deployment",
        "payment",
        "database",
        "db",
        "retry",
    ]

    for term in known_terms:
        if term in lowered:
            keywords.append(term)

    service_or_component = "unknown"
    service_hints = [
        "payment",
        "worker",
        "redis",
        "kafka",
        "database",
        "db",
    ]

    for hint in service_hints:
        if hint in lowered:
            service_or_component = hint
            break

    symptoms: List[str] = []
    if "crashloopbackoff" in lowered:
        symptoms.append("container crash loop")
    if "timeout" in lowered or "deadline exceeded" in lowered:
        symptoms.append("timeout")
    if "latency" in lowered:
        symptoms.append("high latency")

    analysis = {
        "summary": incident,
        "symptoms": symptoms,
        "service_or_component": service_or_component,
        "keywords": keywords,
    }

    return {
        "analysis": analysis,
    }


def rewrite_query_node(state: IncidentState) -> Dict[str, Any]:
    """
    Node 2: Rewrite the incident into a retrieval-optimized query.
    LLM call #1
    """
    incident = state.get("incident", "").strip()
    if not incident:
        raise ValueError("incident cannot be empty")

    analysis = state.get("analysis", {})
    summary = analysis.get("summary", "")
    symptoms = analysis.get("symptoms", [])
    service_or_component = analysis.get("service_or_component", "unknown")
    keywords = analysis.get("keywords", [])

    prompt = f"""
You are a query rewriting assistant for incident retrieval.

Your task:
Rewrite the incident into a compact search query optimized for hybrid retrieval
(keyword search + semantic search).

Rules:
- return only the rewritten query as plain text
- no bullets
- no explanations
- keep it concise but information dense
- preserve exact error words if present
- expand with likely operational terms only when strongly implied by the incident
- do not invent unrelated symptoms

Incident:
{incident}

Lightweight analysis summary:
{summary}

Symptoms:
{symptoms}

Service or component:
{service_or_component}

Keywords:
{keywords}
""".strip()

    rewritten_query = _generate_text(prompt).strip()
    if not rewritten_query:
        raise ValueError("rewrite_query_node produced an empty query")

    return {
        "rewritten_query": rewritten_query,
    }


def incident_search_node(state: IncidentState) -> Dict[str, Any]:
    """
    Node 3: Search the incident knowledge base.
    No LLM call.
    """
    rewritten_query = state.get("rewritten_query", "").strip()
    if not rewritten_query:
        raise ValueError("rewritten_query cannot be empty")

    results = hybrid_search_with_rerank(
        query=rewritten_query,
        dense_limit=10,
        keyword_limit=10,
        hybrid_limit=10,
        final_limit=5,
    )

    if not isinstance(results, list):
        raise ValueError("hybrid_search_with_rerank did not return a list")

    return {
        "retrieved_docs": results,
    }


def triage_planning_node(state: IncidentState) -> Dict[str, Any]:
    """
    Node 4: Generate the final grounded triage plan.
    LLM call #2
    """
    incident = state.get("incident", "").strip()
    if not incident:
        raise ValueError("incident cannot be empty")

    analysis = state.get("analysis", {})
    rewritten_query = state.get("rewritten_query", "").strip()
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        raise ValueError("retrieved_docs is empty; cannot build triage plan")

    context = _build_context(retrieved_docs)

    prompt = f"""
You are an incident triage planning assistant.

Use only the retrieved context below.
Do not invent evidence.
If the evidence is limited, provide the best grounded triage plan possible and stay conservative.

Return ONLY valid JSON in this exact structure:
{{
  "incident_type": "short label",
  "hypotheses": ["hypothesis 1", "hypothesis 2", "hypothesis 3"],
  "next_steps": ["step 1", "step 2", "step 3"],
  "evidence": [
    {{
      "source": "filename or source name",
      "reason": "why this source supports the triage plan"
    }}
  ]
}}

Incident:
{incident}

Analysis:
{json.dumps(analysis, ensure_ascii=False)}

Rewritten query:
{rewritten_query}

Retrieved context:
{context}
""".strip()

    raw_output = _generate_text(prompt)
    parsed = _extract_json_object(raw_output)

    incident_type = str(parsed.get("incident_type", "")).strip()
    hypotheses = _safe_list_of_strings(parsed.get("hypotheses"))
    next_steps = _safe_list_of_strings(parsed.get("next_steps"))

    raw_evidence = parsed.get("evidence", [])
    evidence: List[Dict[str, str]] = []

    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source", "")).strip()
            reason = str(item.get("reason", "")).strip()

            if source and reason:
                evidence.append(
                    {
                        "source": source,
                        "reason": reason,
                    }
                )

    return {
        "incident_type": incident_type,
        "hypotheses": hypotheses,
        "next_steps": next_steps,
        "evidence": evidence,
    }