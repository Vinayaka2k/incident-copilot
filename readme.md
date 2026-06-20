# Signalsly (IncidentCopilot)

Signalsly is an **AI-powered incident triage assistant** designed to halve Mean Time to Resolution (MTTR). By integrating directly with monitoring alerts and PagerDuty webhooks, it instantly retrieves relevant runbooks and past incidents to generate grounded, actionable debugging plans for engineers.

---

## 🚀 Key Features

* **PagerDuty Webhook Integration:** Live webhook ingestion automatically triggers AI triage on incoming alerts.
* **Hybrid Retrieval (Dense + Keyword):** Combines k-NN semantic search and BM25 keyword matching via Amazon OpenSearch Serverless.
* **Precision Reranking:** Applies cross-encoder reranking (Amazon Rerank v1 via Bedrock) to top retrieval candidates to maximize precision.
* **Agentic ReAct Workflow:** Uses a LangGraph-driven loop that parses incidents, rewrites queries, performs hybrid searches, and structures findings.
* **Grounded & Auditable:** Hypotheses are strictly mapped to ingested runbooks and historical postmortems to eliminate hallucinations.

---

## 📊 System Architecture & Tech Stack

```
PagerDuty Alert → FastAPI Webhook → Query Rewrite Node → Hybrid Search Node
                                                               ↓
Grounded Triage Report ← Structured JSON Output ← Agent Reasoning Node (ReAct)
```

| Component | Technology |
|---|---|
| **LLM** | Claude (Anthropic) via Amazon Bedrock |
| **Embeddings** | Amazon Titan Embed Text v2 (1024 dims) |
| **Reranker** | Amazon Rerank v1 (via Bedrock Agent Runtime) |
| **Vector DB / Keyword Search** | Amazon OpenSearch Serverless (k-NN + BM25) |
| **Agent Framework** | LangGraph (StateGraph ReAct Loop) |
| **API / Ingestion** | FastAPI & PagerDuty Webhooks |

---

## 📂 Project Structure

```
incident-copilot/
├── agent/            # LangGraph architecture (graph.py, nodes.py)
├── retrieval/        # Hybrid search & rerank (dense, keyword, reranker)
├── ingestion/        # Document loaders, chunkers, and embedding pipeline
├── db/               # OpenSearch database indexers
├── api/              # FastAPI application server
├── data/             # Source runbooks (.md) and historical incidents
└── tests/            # Test suite and SRE Judge evaluation harness
```

---

## 🛠️ Quick Start

### 1. Prerequisites
* AWS Account with Bedrock & OpenSearch Serverless access.
* Python 3.10+ and a configured PagerDuty account.

### 2. Configuration
Configure your local environment or AWS credentials (`~/.aws/credentials`):
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
region = us-east-1
```

### 3. Setup & Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Ingest and index runbooks
python ingestion/loader.py
python ingestion/chunker.py
python ingestion/embedder.py
python db/opensearch_indexer.py

# Test retrieval pipeline
python retrieval/hybrid_search.py

# Launch FastAPI web server
uvicorn api.main:app --reload
```

---

## 🧪 Automated Evaluation & Reliability

The framework integrates a continuous testing suite leveraging `pytest` with an independent **LLM-as-a-Judge** scoring model (`gemini-2.5-flash`):
* **Evaluation Metrics:** Every generated triage response is systematically graded on a 1–5 rubric for **Groundedness** and **Actionability**.
* **API Resiliency:** API integrations are guarded with `tenacity`-driven exponential backoff with randomized jitter to handle rate limit thresholds (429s).
* **Fault-Tolerant Fallbacks:** Graph search nodes are wrapped in error handles to allow the agent to degrade gracefully using its pre-trained system knowledge if the search service is offline.

---

## 🛡️ Input & Output Guardrails

To prevent prompt injections via untrusted logs and suggestions of dangerous production actions (e.g., system wipes), Signalsly incorporates:
1. **AWS Bedrock Guardrails:** Sanitizes inputs and monitors outputs for malicious instructions, sensitive data leakage, or denied topics.
2. **Contextual Grounding Checks:** Re-validates final responses against retrieved runbooks to block ungrounded advice.
3. **Structured Schemas:** Forced tool use binds agent outputs to strict JSON schemas, ensuring complete traceability.
