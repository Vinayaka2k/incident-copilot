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

Examples:

- Keyword search for exact errors (`CrashLoopBackOff`, `TimeoutError`)
- Semantic search for symptom descriptions ("service is slow after deploy")
- Hybrid search for mixed queries combining both

### Cross-Encoder Reranking (Amazon Rerank v1)

After hybrid retrieval, candidates are reranked using **Amazon Rerank v1** via Bedrock to maximize precision.

- Stage 1: Hybrid retrieval for **recall** (cast a wide net)
- Stage 2: Cross-encoder reranking for **precision** (pick the best results)

### Incident Knowledge Base

Indexes operational documents such as:

- Runbooks (step-by-step debugging guides)
- Incident postmortems (past incidents with root cause and resolution)
- Architecture documentation

### Agentic Triage Workflow

The system uses a lightweight agent workflow to:

1. Analyze the incident description
2. Rewrite the query for optimal retrieval
3. Retrieve relevant knowledge via hybrid search + rerank
4. Generate a structured triage plan using Claude on Bedrock

### Grounded Responses

Every recommendation is backed by **runbook sections or past incidents**, reducing hallucinations and ensuring engineers can verify the guidance.

---

## System Architecture
Incident Description
        │
        ▼
Query Analysis (Claude via Bedrock)
        │
        ▼
Query Rewrite Tool
        │
        ▼
Incident Search Tool
        │
        ├── Dense Search (Titan Embed v2 → OpenSearch KNN)
        ├── Keyword Search (OpenSearch BM25)
        ├── Hybrid Fusion (Reciprocal Rank Fusion)
        └── Reranking (Amazon Rerank v1 via Bedrock)
        │
        ▼
Triage Planning Tool (Claude via Bedrock)
        │
        ▼
Structured Incident Response


---

## AWS Bedrock Integration

All AI capabilities are powered by **Amazon Bedrock** — no self-hosted models, no GPU management.


---

## Tech Stack

| Component          | Technology                                  |
| ------------------ | ------------------------------------------- |
| Embedding Model    | Amazon Titan Embed Text v2 (1024 dims)      |
| Reranker           | Amazon Rerank v1 (via Bedrock Agent Runtime) |
| LLM                | Claude (Anthropic) via Amazon Bedrock        |
| Vector + Text Store| Amazon OpenSearch Serverless                 |
| Dense Search       | OpenSearch KNN (HNSW/FAISS)                 |
| Keyword Search     | OpenSearch BM25 (built-in)                  |
| Hybrid Fusion      | Reciprocal Rank Fusion (RRF)                |
| Agent Framework    | LangGraph                                   |
| API Layer          | FastAPI                                     |
| Cloud Platform     | AWS (Bedrock + OpenSearch Serverless)        |

---

## Project Structure
incident-copilot/
├── agent/
│ ├── graph.py # LangGraph agent workflow
│ └── nodes.py # Agent node definitions
├── retrieval/
│ ├── dense_search.py # Vector search via OpenSearch KNN
│ ├── keyword_search.py # BM25 search via OpenSearch match
│ ├── hybrid_search.py # RRF fusion of dense + keyword
│ └── reranker.py # Amazon Rerank v1 via Bedrock
├── ingestion/
│ ├── loader.py # Load markdown files
│ ├── chunker.py # Chunk documents
│ └── embedder.py # Embed chunks via Titan Embed v2
├── db/
│ └── opensearch_indexer.py # Index embedded chunks into OpenSearch
├── api/
│ └── main.py # FastAPI server
├── data/
│ ├── runbooks/ # Operational runbooks (.md)
│ └── incidents/ # Past incident reports (.md)
├── processed/
│ ├── chunks.json # Chunked documents
│ └── embedded_chunks.json # Chunks with Titan embeddings
├── requirements.txt
└── README.md


---

## Example Dataset

The system ingests operational knowledge such as:
data/
├── runbooks/
│ ├── payment_timeout.md
│ ├── redis_latency.md
│ └── kafka_consumer_lag.md
└── incidents/
├── inc_1023_payment_timeout.md
├── inc_1188_retry_storm.md
└── inc_1342_worker_crashloop.md


Each document is:

1. **Loaded** from markdown files
2. **Chunked** into smaller segments
3. **Embedded** using Amazon Titan Embed Text v2 (1024 dimensions)
4. **Indexed** into Amazon OpenSearch Serverless (both vector and text indexes)

---

## Running the Project

### Prerequisites

- AWS account with Bedrock model access enabled
- AWS credentials configured (`~/.aws/credentials`)
- Amazon OpenSearch Serverless collection created
- Python 3.10+

### 1. Configure AWS Credentials

Create `~/.aws/credentials`:


[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY

Create ~/.aws/config:
[default]
region = us-east-1

### 2. Enable Bedrock Model Access
Go to AWS Console → Bedrock → Model access and request access for:

amazon.titan-embed-text-v2:0
amazon.rerank-v1:0
Claude model (Anthropic)

### 3. Create OpenSearch Serverless Collection
AWS Console → OpenSearch → Serverless → Create Collection
Type: Vector search
Name: incident-copilot
Create encryption, network, and data access policies
Note down the endpoint URL

### 4. Install Dependencies
pip install -r requirements.txt

### 5. Ingest Documents
#### Load and chunk documents
python ingestion/loader.py
python ingestion/chunker.py

#### Embed chunks using Titan Embed v2
python ingestion/embedder.py

#### Index into OpenSearch Serverless
python db/opensearch_indexer.py

### 6. Test Retrieval
#### Test dense search
python retrieval/dense_search.py

#### Test keyword search
python retrieval/keyword_search.py

#### Test hybrid search with reranking
python retrieval/reranker.py

### 7. Start the API
uvicorn api.main:app --reload

### API Example
POST /triage

Request:
{
  "incident": "CrashLoopBackOff payment worker after deploy"
}

Response:
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
    "Verify resource limits and memory usage",
    "Check if rollback resolves the issue"
  ],
  "evidence": [
    "Runbook: worker_crashloop.md",
    "Incident: INC-1342_worker_crashloop"
  ]
}

### How Retrieval Works
Dense Search (Semantic)

User Query
    │
    ▼
Embed via Titan Embed v2 (1024-dim vector)
    │
    ▼
OpenSearch KNN query on "embedding" field
    │
    ▼
Returns top-K semantically similar chunks

### Keyword Search (BM25)

User Query
    │
    ▼
OpenSearch "match" query on "text" field
    │
    ▼
Built-in BM25 tokenization + scoring
    │
    ▼
Returns top-K keyword-matched chunks

### Hybrid Search + Rerank

User Query
    │
    ├──► Dense Search (top 10)
    │
    ├──► Keyword Search (top 10)
    │
    ▼
Reciprocal Rank Fusion (merge + deduplicate)
    │
    ▼
Amazon Rerank v1 (cross-encoder reranking)
    │
    ▼
Final top-5 most relevant chunks

### Why This Project
Many companies already store operational knowledge in runbooks and incident reports, but finding the right information during incidents is slow.

IncidentCopilot demonstrates how AI agents combined with hybrid retrieval on AWS Bedrock can transform operational knowledge into actionable guidance — using a fully managed, production-grade cloud stack.

### Future Improvements
Observability integration (CloudWatch logs / metrics)
Slack or PagerDuty integration
Automatic incident classification
Feedback loop for improving retrieval quality
Bedrock Guardrails for content filtering
Provisioned throughput for cost optimization
Caching embeddings to reduce Bedrock API calls

### Appendix: Migration from Original Stack
This project was initially built with open-source / local models, then migrated entirely to AWS Bedrock and OpenSearch Serverless for a production-grade cloud-native architecture.

#### Before (Local / Mixed)
Gemini (Google) ──── LLM Generation
bge-small-en-v1.5 ── Embedding (local)
MiniLM CrossEncoder ─ Reranking (local)
Qdrant (Docker) ───── Vector Store
rank_bm25 (Python) ── Keyword Search (local)

#### After (AWS Bedrock / OpenSearch)
Claude via Bedrock ────────── LLM Generation
Titan Embed v2 via Bedrock ── Embedding (managed)
Amazon Rerank v1 via Bedrock ─ Reranking (managed)
OpenSearch Serverless ──────── Vector + Keyword Store
OpenSearch BM25 ────────────── Keyword Search (managed)