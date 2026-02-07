"""
Retrieval Module - Hybrid Search, Reranking, and Hierarchical Retrieval.

Consolidates all retrieval components:
- HybridSearch: Optimized BM25 + Vector search (from agentic.retrieval)
- SemanticCache: Query caching with TTL (from agentic.retrieval)  
- Reranker: Cross-encoder + ColBERT reranking (from agentic.retrieval)
- HierarchicalRetriever: Multi-level document retrieval
- SOTA components: ColBERT, Multi-Query RAG, Adaptive Retrieval
"""

# Core retrieval components (optimized versions from agentic.retrieval)
from src.agentic.retrieval.hybrid_search import HybridSearch
from src.agentic.retrieval.semantic_cache import SemanticCache
from src.agentic.retrieval.reranker import Reranker

# Hierarchical retrieval
from src.retrieval.hierarchical_retriever import HierarchicalRetriever

# SOTA enhancements (lazy imports to avoid startup penalty)
def get_colbert_reranker():
    """Get SOTA ColBERT reranker for improved precision."""
    from src.sota.colbert_reranker import get_colbert_reranker
    return get_colbert_reranker()

def get_multi_query_rag():
    """Get SOTA Multi-Query RAG for improved recall."""
    from src.sota.multi_query_rag import MultiQueryRAG
    return MultiQueryRAG()

def get_adaptive_retriever():
    """Get SOTA Adaptive Retriever for cost optimization."""
    from src.sota.adaptive_retrieval import AdaptiveRetriever
    return AdaptiveRetriever()

def get_hybrid_retriever():
    """Get hybrid retriever instance for quotation processing.
    
    Returns a HybridSearch instance configured for product knowledge retrieval.
    """
    return HybridSearch()


__all__ = [
    # Core components
    "HybridSearch",
    "SemanticCache",
    "Reranker",
    "HierarchicalRetriever",
    # SOTA getters
    "get_colbert_reranker",
    "get_multi_query_rag",
    "get_adaptive_retriever",
    "get_hybrid_retriever",
]

