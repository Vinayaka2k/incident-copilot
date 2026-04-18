# IncidentCopilot

**AI-powered incident triage agent built entirely on AWS Bedrock and OpenSearch — helps engineers debug production incidents using runbooks, postmortems, and hybrid retrieval.**

Production incidents are stressful and time-critical. Engineers often waste valuable minutes searching through runbooks, past incident reports, Slack threads, and documentation to determine the next debugging step.

**IncidentCopilot** is an agentic system that automatically analyzes incident descriptions, retrieves relevant operational knowledge, and generates a structured triage plan grounded in real runbooks and past incidents.

The goal is simple: **reduce time-to-context during incidents.**

---

## Demo Example

### Input

    TimeoutError: upstream payment service deadline exceeded

### Output

**Probable Incident Category**  
Payment dependency timeout.

**Most Likely Hypotheses**

1. Downstream Redis or database latency  
2. Retry storm triggered by recent deployment  
3. Connection pool exhaustion in payment worker  

**Recommended Next Steps**

1. Check latest deployment for `payment-service`  
2. Inspect Redis / DB latency dashboards  
3. Verify worker queue backlog and timeout rate  
4. Compare with previous incidents tagged `payment-timeout`  
5. Roll back deployment if latency spike began after release  

**Evidence**

- Runbook: `payment_timeout.md`  
- Incident: `INC-1188_payment_retry_storm`  
- Incident: `INC-1042_redis_latency_spike`  

---

## Key Features

### Hybrid Retrieval (Dense + Keyword via OpenSearch)

Combines **semantic vector search** and **BM25 keyword search** — both powered by Amazon OpenSearch Serverless — to find the most relevant operational knowledge.

- **Dense search**: Embeds the query using Amazon Titan Embed v2 and performs KNN similarity search on OpenSearch  
- **Keyword search**: Uses OpenSearch's built-in BM25 scoring on the `text` field  
- **Hybrid search**: Fuses both result sets using Reciprocal Rank Fusion (RRF)  

**Examples:**

- Keyword search for exact errors (`CrashLoopBackOff`, `TimeoutError`)  
- Semantic search for symptom descriptions ("service is slow after deploy")  
- Hybrid search for mixed queries combining both  

---

### Cross-Encoder Reranking (Amazon Rerank v1)

After hybrid retrieval, candidates are reranked using **Amazon Rerank v1** via Bedrock to maximize precision.

- Stage 1: Hybrid retrieval for **recall**  
- Stage 2: Cross-encoder reranking for **precision**  

---

### Incident Knowledge Base

Indexes operational documents such as:

- Runbooks (step-by-step debugging guides)  
- Incident postmortems  
- Architecture documentation  

---

### Agentic Triage Workflow

1. Analyze the incident description  
2. Rewrite the query  
3. Retrieve relevant knowledge (hybrid + rerank)  
4. Generate a structured triage plan  

---

### Grounded Responses

Every recommendation is backed by **runbook sections or past incidents**, reducing hallucinations.

---

## System Architecture

**Flow (linearized for GitHub compatibility):**

1. Incident Description  
2. → Query Analysis (Claude via Bedrock)  
3. → Query Rewrite Tool  
4. → Incident Search Tool  
   - Dense Search (Titan Embed v2 → OpenSearch KNN)  
   - Keyword Search (OpenSearch BM25)  
   - Hybrid Fusion (RRF)  
   - Reranking (Amazon Rerank v1)  
5. → Triage Planning Tool (Claude via Bedrock)  
6. → Structured Incident Response  

---

## AWS Bedrock Integration

All AI capabilities are powered by **Amazon Bedrock** — no self-hosted models, no GPU management.

---

## Tech Stack

| Component           | Technology                                      |
|--------------------|--------------------------------------------------|
| Embedding Model     | Amazon Titan Embed Text v2 (1024 dims)          |
| Reranker            | Amazon Rerank v1 (via Bedrock Agent Runtime)    |
| LLM                 | Claude (Anthropic) via Amazon Bedrock           |
| Vector + Text Store | Amazon OpenSearch Serverless                    |
| Dense Search        | OpenSearch KNN                                  |
| Keyword Search      | OpenSearch BM25                                 |
| Hybrid Fusion       | Reciprocal Rank Fusion                          |
| Agent Framework     | LangGraph                                       |
| API Layer           | FastAPI                                         |
| Cloud Platform      | AWS                                             |

---

## Project Structure

    incident-copilot/
    ├── agent/
    │   ├── graph.py
    │   └── nodes.py
    ├── retrieval/
    │   ├── dense_search.py
    │   ├── keyword_search.py
    │   ├── hybrid_search.py
    │   └── reranker.py
    ├── ingestion/
    │   ├── loader.py
    │   ├── chunker.py
    │   └── embedder.py
    ├── db/
    │   └── opensearch_indexer.py
    ├── api/
    │   └── main.py
    ├── data/
    │   ├── runbooks/
    │   └── incidents/
    ├── processed/
    │   ├── chunks.json
    │   └── embedded_chunks.json
    ├── requirements.txt
    └── README.md

---

## Example Dataset

    data/
    ├── runbooks/
    │   ├── payment_timeout.md
    │   ├── redis_latency.md
    │   └── kafka_consumer_lag.md
    └── incidents/
        ├── inc_1023_payment_timeout.md
        ├── inc_1188_retry_storm.md
        └── inc_1342_worker_crashloop.md

Each document is:

1. Loaded  
2. Chunked  
3. Embedded (Titan Embed v2)  
4. Indexed into OpenSearch  

---

## Running the Project

### Prerequisites

- AWS account with Bedrock access  
- AWS credentials configured  
- OpenSearch Serverless collection  
- Python 3.10+  

---

### 1. Configure AWS Credentials

    [default]
    aws_access_key_id = YOUR_ACCESS_KEY
    aws_secret_access_key = YOUR_SECRET_KEY

    [default]
    region = us-east-1

---

### 2. Enable Bedrock Models

Enable:

- `amazon.titan-embed-text-v2:0`  
- `amazon.rerank-v1:0`  
- Claude (Anthropic)  

---

### 3. Create OpenSearch Collection

- Type: Vector search  
- Name: `incident-copilot`  
- Save endpoint  

---

### 4. Install Dependencies

    pip install -r requirements.txt

---

### 5. Ingest Documents

    python ingestion/loader.py
    python ingestion/chunker.py
    python ingestion/embedder.py
    python db/opensearch_indexer.py

---

### 6. Test Retrieval

    python retrieval/dense_search.py
    python retrieval/keyword_search.py
    python retrieval/reranker.py

---

### 7. Start API

    uvicorn api.main:app --reload

---

### API Example

**POST /triage**

**Request**

```json
{
  "incident": "CrashLoopBackOff payment worker after deploy"
}
```


**Response**
{
  "incident_type": "container crash loop",
  "hypotheses": [
    "Bad configuration in latest deployment",
    "OOM kill due to memory limit",
    "Missing environment variable or secret"
  ],
  "next_steps": [
    "Check deployment diff for payment-worker",
    "Inspect pod logs and events",
    "Verify resource limits",
    "Check rollback"
  ],
  "evidence": [
    "Runbook: worker_crashloop.md",
    "Incident: INC-1342_worker_crashloop"
  ]
}

## How Retrieval Works

### Dense Search

- Input query  
- → Embed (Titan)  
- → KNN search  
- → Top-K semantic matches  

---

### Keyword Search

- Input query  
- → BM25 match  
- → Top-K keyword matches  

---

### Hybrid + Rerank

1. Dense results (top 10)  
2. Keyword results (top 10)  
3. Merge via RRF  
4. Rerank (Amazon Rerank v1)  
5. Return top 5  

---

## Why This Project

Operational knowledge exists but is hard to use during incidents.

IncidentCopilot turns it into **actionable guidance** using AI + hybrid retrieval.

---

## Future Improvements

- Observability integration  
- Slack / PagerDuty integration  
- Auto classification  
- Feedback loop  
- Guardrails  
- Cost optimization  
- Embedding cache  

---

## Appendix: Migration

### Before

- Gemini → LLM  
- bge-small → embeddings  
- MiniLM → reranking  
- Qdrant → vector DB  
- rank_bm25 → keyword  

---

### After

- Claude → LLM  
- Titan → embeddings  
- Amazon Rerank → reranking  
- OpenSearch → vector + keyword  
- BM25 → keyword scoring  

# How to Add Guardrails to the Incident Copilot System Using Bedrock

## Why Guardrails Are Critical for This System

In an incident copilot system using RAG, the model consumes retrieved logs, runbooks, and incident data that may be **untrusted or poisoned**. This can lead to:

- Prompt injection via retrieved content  
- Hallucinated or irrelevant conclusions  
- Suggestions of **dangerous production actions** (e.g., delete, restart, wipe systems)  

Since this system recommends **next steps during live incident triage**, unsafe outputs can cause **real operational damage**.

Guardrails ensure the model:
- Ignores malicious input  
- Follows strict reasoning rules  
- Produces structured, auditable outputs  
- Does not suggest harmful actions  

---

## 1. Guardrails to Sanitize Input (Retrieval Sanitization)

### Why this is needed

Retrieved documents are **not trustworthy** and may contain hidden prompt injections or malicious instructions. These must not influence the model.

### Bedrock Features Used

- **Guardrails → Content Filters (Prompt Attack – Input)**  
- **Guardrails → Word Filters**  
- **Guardrails → Denied Topics**

### Why these features

They scan all input **before it reaches the model**, blocking injection attempts and unsafe intent early.

---

## 2. System Prompts for Claude Behavior Control

### Why this is needed

The model must:
- Treat retrieved data as **untrusted evidence**
- Avoid destructive actions  
- Focus on **diagnosis, not execution**  
- Base all claims on evidence  

### Bedrock Features Used

- **Converse API → System Prompt**  
- **(Optional) Bedrock Agents → Instructions**

### Why these features

They enforce **consistent reasoning behavior** and strong safety constraints at the model level.

---

## 3. Restrict Output to Structured JSON (Tool Use)

### Why this is needed

Structured output ensures:
- Traceable claims with evidence  
- Explicit identification of risks  
- Easy validation and downstream use  

### Bedrock Features Used

- **Converse API → Tool Use (`toolConfig`)**  
- **Forced Tool Selection (`toolChoice`)**

### Why these features

They ensure the model **always returns structured, schema-aligned output** instead of free text.

---

## 4. Output Filtering Layer (Safety Enforcement)

### Why this is needed

Even after generation, the model may:
- Suggest risky or destructive actions  
- Produce unsafe or irrelevant responses  
- Include sensitive information  

### Bedrock Features Used

- **Guardrails → Content Filters (Output)**  
- **Guardrails → Denied Topics (Output)**  
- **Guardrails → Sensitive Information Filters**  
- **Guardrails → Contextual Grounding Check**

### Why these features

They validate the **final response before delivery**, ensuring:
- No harmful suggestions reach engineers  
- Responses are grounded in retrieved evidence  
- Sensitive data is blocked or masked  

---