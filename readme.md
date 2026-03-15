# IncidentCopilot

**AI agent that helps engineers triage production incidents using runbooks, postmortems, and hybrid retrieval.**

Production incidents are stressful and time-critical. Engineers often waste valuable minutes searching through runbooks, past incident reports, Slack threads, and documentation to determine the next debugging step.

**IncidentCopilot** is an agentic system that automatically analyzes incident descriptions, retrieves relevant operational knowledge, and generates a structured triage plan grounded in real runbooks and past incidents.

The goal is simple: **reduce time-to-context during incidents.**

---

# Demo Example

### Input

```
TimeoutError: upstream payment service deadline exceeded
```

### Output

**Probable Incident Category**
Payment dependency timeout.

**Most Likely Hypotheses**

1. Downstream Redis or database latency.
2. Retry storm triggered by recent deployment.
3. Connection pool exhaustion in payment worker.

**Recommended Next Steps**

1. Check latest deployment for `payment-service`.
2. Inspect Redis / DB latency dashboards.
3. Verify worker queue backlog and timeout rate.
4. Compare with previous incidents tagged `payment-timeout`.
5. Roll back deployment if latency spike began after release.

**Evidence**

* Runbook: `payment_timeout.md`
* Incident: `INC-1188_payment_retry_storm`
* Incident: `INC-1042_redis_latency_spike`

---

# Key Features

### Hybrid Retrieval

Combines **semantic search** and **keyword search** to find the most relevant operational knowledge.

Examples:

* Keyword search for exact errors (`CrashLoopBackOff`, `TimeoutError`)
* Semantic search for symptom descriptions
* Hybrid search for mixed queries

### Incident Knowledge Base

Indexes operational documents such as:

* Runbooks
* Incident postmortems
* Architecture documentation

### Agentic Triage Workflow

The system uses a lightweight agent workflow to:

1. Analyze the incident description
2. Rewrite the query for optimal retrieval
3. Retrieve relevant knowledge
4. Generate a structured triage plan

### Grounded Responses

Every recommendation is backed by **runbook sections or past incidents**, reducing hallucinations.

---

# System Architecture

```
Incident description
        │
        ▼
Query Analysis
        │
        ▼
Query Rewrite Tool
        │
        ▼
Incident Search Tool
(Dense + Sparse + Hybrid Retrieval via Qdrant)
        │
        ▼
Triage Planning Tool
        │
        ▼
Structured Incident Response
```

---

# Tech Stack

| Component       | Technology                            |
| --------------- | ------------------------------------- |
| LLM             | OpenAI / LLM API                      |
| Vector Database | Qdrant                                |
| Agent Framework | LangGraph                             |
| API Layer       | FastAPI                               |
| Embeddings      | Sentence Transformers / OpenAI        |
| Search          | Semantic + Keyword + Hybrid Retrieval |

---

# Project Structure

```
incident-copilot/

agent/
    graph.py
    nodes.py

retrieval/
    dense_search.py
    sparse_search.py
    hybrid_search.py

ingestion/
    loader.py
    chunker.py
    embedder.py

api/
    main.py

data/
    runbooks/
    incidents/

README.md
```

---

# Example Dataset

The system ingests operational knowledge such as:

```
data/

runbooks/
    payment_timeout.md
    redis_latency.md
    kafka_consumer_lag.md

incidents/
    inc_1023_payment_timeout.md
    inc_1188_retry_storm.md
    inc_1342_worker_crashloop.md
```

Each document is chunked, embedded, and indexed in Qdrant.

---

# Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Start Qdrant

```
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Ingest documents

```
python ingestion/pipeline.py
```

### 4. Start the API

```
uvicorn api.main:app --reload
```

---

# API Example

```
POST /triage
```

Request:

```
{
  "incident": "CrashLoopBackOff payment worker after deploy"
}
```

Response:

```
{
  "incident_type": "container crash loop",
  "hypotheses": [...],
  "next_steps": [...],
  "evidence": [...]
}
```

---

# Why This Project

Many companies already store operational knowledge in runbooks and incident reports, but **finding the right information during incidents is slow**.

IncidentCopilot demonstrates how **AI agents combined with hybrid retrieval** can transform operational knowledge into actionable guidance.

---

# Future Improvements

* Observability integration (logs / metrics)
* Slack or PagerDuty integration
* Automatic incident classification
* Feedback loop for improving retrieval quality

---

# License

MIT
