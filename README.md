# JD Jones RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with Agentic AI capabilities, hierarchical knowledge bases, Super Memory integration, and multi-provider context synchronization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          JD JONES RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       AGENTIC AI LAYER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Orchestrator │  │ ReAct Agent │  │ Multi-Agent │  │ Guardrails │  │   │
│  │  │   (Brain)    │  │  (Reason+   │  │ Coordinator │  │  (Safety)  │  │   │
│  │  │              │  │    Act)     │  │             │  │            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Product    │  │  Enquiry    │  │  Validation │  │   HITL     │  │   │
│  │  │  Selection  │  │  Management │  │    Agent    │  │ Approvals  │  │   │
│  │  │   Agent     │  │    Agent    │  │             │  │            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    KNOWLEDGE BASE HIERARCHY                          │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ LEVEL 0 (Main Context) - Company-Wide Knowledge             │   │   │
│  │  │ o Product Catalog o Policies o Specifications o FAQs        │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ LEVEL 1 (Department Contexts) - Role-Based Access           │   │   │
│  │  │ o Sales o Production o Engineering o Customer Service       │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   ADVANCED RETRIEVAL LAYER                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Hybrid Search│  │  Re-ranker   │  │   Semantic Cache         │  │   │
│  │  │ (BM25+Vector)│  │  (LLM-based) │  │   (Redis-backed)         │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     SUPER MEMORY SYSTEM                              │   │
│  │  . PostgreSQL + pgvector for persistent memory storage              │   │
│  │  . Runtime context loading for personalized responses               │   │
│  │  . Auto-learning from conversations                                  │   │
│  │  . Multi-provider memory sync (Claude, OpenAI, Gemini)              │   │
│  │  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │   INTERNAL AGENT     │    │   EXTERNAL SYSTEM    │                      │
│  │   (Employees)        │    │   (Customers)        │                      │
│  │   . Conversational   │    │   . Decision Tree    │                      │
│  │   . Access-Controlled│    │   . Guided Journey   │                      │
│  │   . Memory-Enhanced  │    │   . Form Collection  │                      │
│  └──────────────────────┘    └──────────────────────┘                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      OBSERVABILITY LAYER                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Agent Tracer │  │   Monitor    │  │   Alert Management       │  │   │
│  │  │ (LangSmith)  │  │  (Metrics)   │  │   (Thresholds)          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agentic AI Capabilities

### Why Agentic AI?
Standard RAG is a "one-shot" system - query -> retrieve -> answer. Agentic RAG introduces a reasoning loop that allows the system to think, plan, act, and iterate.

### Key Features

| Feature | Description |
|---------|-------------|
| **ReAct Agents** | Reason + Act loop for iterative problem solving with self-correction |
| **Multi-Agent Coordination** | Specialized agents (Researcher, Writer, Reviewer, Executor) working together |
| **Guided Product Selection** | Decision tree that asks targeted questions about industry, equipment, temperature, pressure |
| **Enquiry Management** | Auto-classify, route, and respond to customer enquiries |
| **Human-in-the-Loop** | Approval workflows for sensitive actions (emails, financial, legal) |
| **Guardrails** | PII detection, prompt injection prevention, content policy enforcement |
| **Hybrid Search** | BM25 + Vector search for better precision on technical terminology |
| **Semantic Caching** | Reduce latency and costs for high-frequency queries |
| **Observability** | Full tracing of agent reasoning, tool calls, and retrievals |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API Key

### Installation

1. **Clone and setup:**
```bash
git clone https://github.com/JdJonesTech/Rag-Knowledge-System.git
cd Rag-Knowledge-System/jd_jones_rag
cp .env.example .env
```

2. **Configure environment:**
Edit `.env` and add your API keys:
```bash
OPENAI_API_KEY=sk-your-key-here
POSTGRES_PASSWORD=your-secure-password
JWT_SECRET_KEY=your-jwt-secret
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Initialize database:**
```bash
docker exec -it jd_jones_postgres psql -U jdjones -d jd_jones_rag -f /docker-entrypoint-initdb.d/01_schema.sql
```

5. **Ingest documents:**
```bash
docker exec -it jd_jones_api python scripts/ingest_documents.py --source /app/documents --level main
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| Internal Portal | http://localhost:3000 | Employee chatbot UI |
| External Portal | http://localhost:3001 | Customer decision tree UI |
| Flower | http://localhost:5555 | Celery monitoring |

## Project Documentation

Detailed documentation is available for collaborators:

- [Collaboration Guide](COLLABORATION.md) - How to set up and contribute.
- [Contributing Guidelines](CONTRIBUTING.md) - Workflow and PR policy.

## Security

- JWT-based authentication with configurable expiration
- Role-based access control for knowledge bases
- Encrypted sensitive data in environment variables
- CORS configuration for frontend applications

