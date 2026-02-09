# JD Jones RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with **Agentic AI capabilities**, hierarchical knowledge bases, Super Memory integration, and multi-provider context synchronization.

## 🏗️ Architecture

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
│  │  │ • Product Catalog • Policies • Specifications • FAQs        │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ LEVEL 1 (Department Contexts) - Role-Based Access           │   │   │
│  │  │ • Sales • Production • Engineering • Customer Service       │   │   │
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
│  │  • PostgreSQL + pgvector for persistent memory storage              │   │
│  │  • Runtime context loading for personalized responses               │   │
│  │  • Auto-learning from conversations                                  │   │
│  │  • Multi-provider memory sync (Claude, OpenAI, Gemini)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │   INTERNAL AGENT     │    │   EXTERNAL SYSTEM    │                      │
│  │   (Employees)        │    │   (Customers)        │                      │
│  │   • Conversational   │    │   • Decision Tree    │                      │
│  │   • Access-Controlled│    │   • Guided Journey   │                      │
│  │   • Memory-Enhanced  │    │   • Form Collection  │                      │
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
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🤖 Agentic AI Capabilities

### Why Agentic AI?
Standard RAG is a "one-shot" system - query → retrieve → answer. Agentic RAG introduces a **reasoning loop** that allows the system to think, plan, act, and iterate.

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

### Agent Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (Brain)                       │
│  • Analyzes queries and identifies missing parameters             │
│  • Coordinates tools and specialized agents                       │
│  • Validates results against industry standards                   │
│  • Generates accurate, helpful responses                          │
└──────────────────────────────────────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Router Agent   │   │  Query Planner  │   │ Validation Agent│
│  (Intent &      │   │  (Decompose     │   │ (Fact-check &   │
│   Routing)      │   │   Complex Tasks)│   │  Compliance)    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                          TOOLS LAYER                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ Vector   │ │ Product  │ │ ERP/SQL  │ │ CRM      │ │ Email    ││
│  │ Search   │ │ Database │ │ Query    │ │ Update   │ │ Router   ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                          │
│  │Compliance│ │ Document │ │ External │                          │
│  │ Checker  │ │Generator │ │   API    │                          │
│  └──────────┘ └──────────┘ └──────────┘                          │
└───────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- OpenAI API Key

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd jd_jones_rag
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

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| Internal Portal | http://localhost:3000 | Employee chatbot UI |
| External Portal | http://localhost:3001 | Customer decision tree UI |
| Flower | http://localhost:5555 | Celery monitoring |

## 📁 Project Structure

```
jd_jones_rag/
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Application container
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config/                # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py        # Pydantic settings
│   │   └── access_control.yaml # Role-based access
│   │
│   ├── data_ingestion/        # Document processing
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── embedding_generator.py
│   │   └── vector_store.py
│   │
│   ├── knowledge_base/        # RAG knowledge bases
│   │   ├── __init__.py
│   │   ├── main_context.py    # Level 0 (company-wide)
│   │   ├── level_contexts.py  # Level 1+ (departments)
│   │   └── retriever.py       # Hierarchical retrieval
│   │
│   ├── agentic/               # 🤖 AGENTIC AI MODULE
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Central brain/coordinator
│   │   ├── router_agent.py    # Query analysis & routing
│   │   ├── reflection_loop.py # Self-correction & validation
│   │   │
│   │   ├── agents/            # Specialized agents
│   │   │   ├── __init__.py
│   │   │   ├── react_agent.py          # ReAct (Reason+Act) agent
│   │   │   ├── query_planner.py        # Complex task decomposition
│   │   │   ├── validation_agent.py     # Fact-checking agent
│   │   │   ├── product_selection_agent.py  # Guided product selection
│   │   │   └── enquiry_management_agent.py # Enquiry classification
│   │   │
│   │   ├── tools/             # Agent tools
│   │   │   ├── __init__.py
│   │   │   ├── base_tool.py
│   │   │   ├── vector_search_tool.py
│   │   │   ├── sql_query_tool.py       # ERP queries
│   │   │   ├── api_tool.py             # External APIs
│   │   │   ├── email_tool.py           # Email routing
│   │   │   ├── crm_tool.py             # CRM operations
│   │   │   ├── document_generator_tool.py
│   │   │   └── compliance_checker_tool.py
│   │   │
│   │   ├── retrieval/         # Advanced retrieval
│   │   │   ├── __init__.py
│   │   │   ├── hybrid_search.py        # BM25 + Vector
│   │   │   ├── reranker.py             # Result re-ranking
│   │   │   └── semantic_cache.py       # Query caching
│   │   │
│   │   ├── multi_agent/       # Multi-agent coordination
│   │   │   ├── __init__.py
│   │   │   ├── coordinator.py
│   │   │   └── agent_registry.py
│   │   │
│   │   ├── memory/            # Agent memory
│   │   │   ├── __init__.py
│   │   │   ├── conversation_memory.py  # Short-term
│   │   │   └── long_term_memory.py     # Persistent
│   │   │
│   │   ├── hitl/              # Human-in-the-loop
│   │   │   ├── __init__.py
│   │   │   ├── approval_manager.py
│   │   │   └── guardrails.py
│   │   │
│   │   └── observability/     # Tracing & monitoring
│   │       ├── __init__.py
│   │       ├── tracer.py
│   │       └── monitor.py
│   │
│   ├── agents/                # AI agents (legacy)
│   │   ├── __init__.py
│   │   ├── internal_agent.py  # Employee chatbot
│   │   └── prompts/
│   │       └── internal_system_prompt.txt
│   │
│   ├── external_system/       # Customer-facing system
│   │   ├── __init__.py
│   │   ├── classifier.py      # Intent classification
│   │   ├── decision_tree.py   # Navigation tree
│   │   └── response_generator.py
│   │
│   ├── auth/                  # Authentication
│   │   ├── __init__.py
│   │   ├── authentication.py  # JWT auth
│   │   └── authorization.py   # RBAC
│   │
│   ├── api/                   # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py           # App entry point
│   │   ├── routers/
│   │   │   ├── internal_chat.py
│   │   │   ├── external_portal.py
│   │   │   └── admin.py
│   │   ├── routes/
│   │   │   └── agentic.py    # 🤖 Agentic API routes
│   │   └── schemas/
│   │       └── requests.py
│   │
│   └── super_memory/          # Super Memory Plugin
│       ├── __init__.py
│       ├── memory_manager.py  # Core memory CRUD
│       ├── context_loader.py  # Runtime loading
│       ├── memory_learner.py  # Auto-learning
│       ├── providers/         # Multi-provider sync
│       │   ├── __init__.py
│       │   ├── base_provider.py
│       │   ├── claude_provider.py
│       │   ├── openai_provider.py
│       │   └── gemini_provider.py
│       └── sync/
│           ├── __init__.py
│           ├── memory_sync_orchestrator.py
│           └── background_sync.py
│
├── frontend/                  # UI applications
│   ├── internal-portal/       # Employee chatbot
│   └── external-portal/       # Customer decision tree
│
├── scripts/                   # Utility scripts
│   ├── ingest_documents.py
│   ├── update_embeddings.py
│   └── sync_database.py
│
├── tests/                     # Test suite
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_agents.py
│   └── test_agentic.py       # 🤖 Agentic AI tests
│
└── sql/                       # Database schemas
    └── super_memory_schema.sql
```

## 🔧 Configuration

### Access Control

Edit `src/config/access_control.yaml` to configure role-based access:

```yaml
roles:
  company_wide:
    - product_catalog
    - company_policies
  sales:
    inherits: company_wide
    additional:
      - pricing_guides
      - customer_database
  production:
    inherits: company_wide
    additional:
      - work_instructions
      - machine_manuals
```

### Super Memory Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `MEMORY_MAX_PER_USER` | Maximum memories per user | 10000 |
| `MEMORY_CACHE_TTL` | Cache time-to-live (seconds) | 3600 |
| `MEMORY_SIMILARITY_THRESHOLD` | Deduplication threshold | 0.92 |
| `AUTO_SYNC_ENABLED` | Enable background sync | true |

## 📖 API Reference

### Internal Chat (Employees)

```bash
# Send message
POST /internal/chat
{
    "message": "What are the shipping specifications for product X?",
    "session_id": "optional-session-id"
}

# Get conversation history
GET /internal/sessions/{session_id}/history
```

### External Portal (Customers)

```bash
# Get decision tree
GET /external/decision-tree

# Navigate to node
POST /external/navigate
{
    "node_id": "product_info",
    "collected_data": {}
}

# Submit form
POST /external/submit-form
{
    "form_type": "quote_request",
    "data": {...}
}
```

### Memory Sync

```bash
# Upload memory export
POST /memory-sync/upload/{provider}
# provider: claude, openai, gemini

# Trigger sync
POST /memory-sync/trigger
{
    "providers": ["claude", "openai"],
    "full_sync": false
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v
```

## 📊 Monitoring

- **Flower Dashboard**: http://localhost:5555 - Monitor Celery tasks
- **API Health**: http://localhost:8000/health - Check API status
- **PostgreSQL**: Use `psql` or pgAdmin to monitor database

## 🔐 Security

- JWT-based authentication with configurable expiration
- Role-based access control for knowledge bases
- Encrypted sensitive data in environment variables
- CORS configuration for frontend applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
