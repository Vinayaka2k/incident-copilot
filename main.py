from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import incident_graph


app = FastAPI(
    title="Incident Copilot API",
    description="FastAPI service for IncidentCopilot using LangGraph + hybrid retrieval + Gemini",
    version="1.0.0",
)


class IncidentRequest(BaseModel):
    incident: str = Field(
        ...,
        min_length=1,
        description="Raw incident description from the user",
    )


class EvidenceItem(BaseModel):
    source: str
    reason: str


class IncidentResponse(BaseModel):
    incident: str
    incident_type: str
    hypotheses: List[str]
    next_steps: List[str]
    evidence: List[EvidenceItem]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=IncidentResponse)
def triage_incident(request: IncidentRequest) -> IncidentResponse:
    try:
        final_state = incident_graph.invoke(
            {
                "incident": request.incident.strip(),
            }
        )

        return IncidentResponse(
            incident=final_state.get("incident", request.incident.strip()),
            incident_type=final_state.get("incident_type", ""),
            hypotheses=final_state.get("hypotheses", []),
            next_steps=final_state.get("next_steps", []),
            evidence=final_state.get("evidence", []),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)