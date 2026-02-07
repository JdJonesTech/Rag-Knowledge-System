"""
Agentic AI Module
Provides reasoning, orchestration, and multi-tool capabilities on top of RAG.

Components:
- Orchestrator: Central brain that coordinates all operations
- Router Agent: Query analysis and routing
- Reflection Loop: Self-correction and validation
- Tools: Vector search, SQL queries, APIs, CRM, email
- Specialized Agents: ReAct, Product Selection, Enquiry Management
- Multi-Agent Coordinator: Complex workflow orchestration
- Human-in-the-Loop: Approvals and guardrails
- Memory: Conversation and long-term memory
- Retrieval: Hybrid search, reranking, caching
- Observability: Tracing and monitoring
- SLMs: SQL generation, PII filtering, fast classification

SOTA Integrations (via orchestrator):
- Tiered Intelligence: LLM → SLM → sklearn routing (70% faster)
- ColBERT Reranking: 5-10x faster precision
- Cache-Augmented Generation: 40x faster common queries
- Multi-Query RAG: 15-20% recall improvement
"""

from src.agentic.orchestrator import AgentOrchestrator
from src.agentic.router_agent import RouterAgent, QueryAnalysis, QueryIntent
from src.agentic.reflection_loop import ReflectionLoop

__all__ = [
    # Core
    "AgentOrchestrator",
    "RouterAgent",
    "QueryAnalysis", 
    "QueryIntent",
    "ReflectionLoop",
    
    # Submodules
    "agents",
    "tools",
    "retrieval",
    "hitl",
    "observability",
    "memory",
    "multi_agent",
    "slm"
]
