import os
import json
import pytest
from typing import Dict, Any, List

# Import your graph under test
from agent.graph import incident_graph
from agent.nodes import get_client

# Define the model to use as the Judge (prefer a strong model for evaluating)
JUDGE_MODEL = "gemini-2.5-flash"

# Define your "Golden Dataset" (Input -> Expected attributes)
GOLDEN_DATASET = [
    {
        "id": "payment_timeout",
        "incident": "TimeoutError: upstream payment service deadline exceeded. High latency on DB queries.",
        "expected_incident_type": "Payment dependency timeout or latency",
        "required_action_keywords": ["database", "redis", "timeout", "rollback", "connection"],
    },
    {
        "id": "redis_spike",
        "incident": "Redis latency spike on cache workers. Users experiencing slow login flows.",
        "expected_incident_type": "Redis or login dependency issue",
        "required_action_keywords": ["cache", "redis", "latency", "connection", "login"],
    }
]


def llm_judge_evaluate(incident: str, output: Dict[str, Any], context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Uses Gemini as a Judge to grade the triage plan's Groundedness and Actionability.
    """
    client = get_client()
    
    # Format the context retrieved during the agent run
    context_str = "\n\n".join([f"Source: {doc.get('source', 'unknown')}\nContent: {doc.get('text', '')}" for doc in context_docs])
    if not context_str:
        context_str = "No retrieved documents."

    prompt = f"""
    You are an expert Systems Reliability Engineering (SRE) Judge evaluating an automated AI incident triage copilot.
    
    [INPUT INCIDENT]
    {incident}
    
    [RETRIEVED RUNBOOKS/CONTEXT]
    {context_str}
    
    [COPILOT RESPONSE]
    {json.dumps(output, indent=2)}
    
    Evaluate the copilot's response based on the following two criteria. Provide a score from 1 to 5 for each.

    1. GROUNDEDNESS & FAITHFULNESS (1 to 5):
       - 5: Every single recommendation and hypothesis is strictly justified and supported by the retrieved runbooks/context. Zero hallucinations.
       - 3: Mostly grounded, but introduces some general or pre-trained assumptions not found in the context.
       - 1: Highly hallucinatory; invents runbook steps or facts not present in the retrieved context.

    2. ACTIONABILITY (1 to 5):
       - 5: Next steps are highly concrete and specific to this system (e.g., checking specific metrics, commands, or dashboards).
       - 3: Next steps are standard generic advice (e.g., "Check logs", "Investigate database").
       - 1: Steps are completely irrelevant, vague, or passive.

    Return your output strictly as a JSON object with this exact structure:
    {{
        "groundedness_score": int,
        "actionability_score": int,
        "groundedness_rationale": "...",
        "actionability_rationale": "..."
    }}
    Do not return any extra markdown formatting or wrappers like ```json, just the raw JSON text.
    """
    
    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt
    )
    
    text = response.text.strip()
    
    # Simple JSON extraction helper to handle raw output or block-wrapped output cleanly
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
        
    try:
        return json.loads(text)
    except Exception as e:
        # Fallback if parsing fails
        return {
            "groundedness_score": 1,
            "actionability_score": 1,
            "error": f"Failed to parse Judge JSON output: {str(e)}. Original text: {text}"
        }


@pytest.mark.parametrize("case", GOLDEN_DATASET, ids=lambda x: x["id"])
def test_agent_evals(case: Dict[str, Any]):
    """
    Evaluates the incident copilot's outputs on target incident queries.
    """
    # 1. Run the actual agent graph under test
    print(f"\n--- Running evaluation for: {case['id']} ---")
    final_state = incident_graph.invoke(
        {
            "incident": case["incident"],
        }
    )
    
    # Extract relevant fields from the final agent state
    final_answer = final_state.get("final_answer", {})
    retrieved_docs = final_state.get("retrieved_docs", [])
    
    hypotheses = final_answer.get("hypothesis", []) or final_answer.get("hypotheses", [])
    next_steps = final_answer.get("next_steps", [])
    
    # 2. Check basic functional invariants (Unit assertions)
    assert len(hypotheses) > 0, "Agent failed to generate any hypotheses."
    assert len(next_steps) > 0, "Agent failed to generate any next steps."
    
    # Verify keyword coverage in next steps
    flat_next_steps = " ".join(next_steps).lower()
    keyword_matches = [kw for kw in case["required_action_keywords"] if kw in flat_next_steps]
    print(f"Keyword Matches Found: {keyword_matches} of {case['required_action_keywords']}")
    
    # 3. Perform Semantic LLM-as-a-Judge Evaluation
    judgement = llm_judge_evaluate(
        incident=case["incident"],
        output=final_answer,
        context_docs=retrieved_docs
    )
    
    print("\n[JUDGE VERDICT]")
    print(json.dumps(judgement, indent=2))
    
    # In enterprise setups, we typically enforce a minimum passing threshold (e.g., 3.5 or 4 out of 5)
    min_passing_score = 3
    
    assert judgement["groundedness_score"] >= min_passing_score, (
        f"Groundedness score is too low ({judgement['groundedness_score']}/5). "
        f"Rationale: {judgement.get('groundedness_rationale')}"
    )
    
    assert judgement["actionability_score"] >= min_passing_score, (
        f"Actionability score is too low ({judgement['actionability_score']}/5). "
        f"Rationale: {judgement.get('actionability_rationale')}"
    )
