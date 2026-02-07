"""
SOTA (State-of-the-Art) RAG Enhancements Module

This module contains implementations of the latest SOTA techniques for RAG systems:

Phase 1 - Quick Wins (~10ms-500ms latency reduction):
1. Response Streaming - Token-by-token output
2. Adaptive Retrieval - Query classification to skip unnecessary retrieval
3. Tiered Intelligence - LLM → SLM → sklearn routing

Phase 2 - Core Improvements (~500ms-2s latency reduction):
4. ColBERTv2 Reranker - Late interaction for 5-10x faster reranking
5. Domain-Adapted Embeddings - Fine-tuned embeddings for 10-15% precision gain
6. Embedding Warmup - Eliminate 2-3s cold start

Phase 3 - Advanced Features:
7. Multi-Query RAG - Query decomposition for 15-20% recall improvement
8. Cache-Augmented Generation - 40x faster for common queries
9. Multi-Hop GraphRAG - Complex multi-relationship queries
10. Speculative Decoding - 2-3x faster LLM inference

Phase 4 - Enterprise Scale:
11. Distributed Search - Horizontal sharding
12. A/B Testing - Experiment framework
13. Rate Limiting - Traffic control
"""

# Phase 1: Quick Wins
from src.sota.response_streaming import (
    StreamingResponseGenerator,
    StreamChunk,
    SSEFormatter,
    get_streamer
)
from src.sota.adaptive_retrieval import (
    AdaptiveRetriever,
    QueryClassification,
    QueryType,
)
from src.sota.tiered_intelligence import (
    TieredIntelligence,
    IntelligenceTier,
    TierDecision,
    TierResponse,
    get_tiered_intelligence
)

# Phase 2: Core Improvements
from src.sota.colbert_reranker import (
    ColBERTReranker,
    ColBERTResult,
    get_colbert_reranker
)
from src.sota.domain_embeddings import (
    DomainAdaptedEmbedder,
    EmbeddingResult,
    TrainingPair,
    get_domain_embedder
)
from src.sota.embedding_warmup import (
    EmbeddingWarmup,
    WarmupStats,
    get_embedding_warmup,
    startup_warmup
)

# Phase 3: Advanced Features
from src.sota.multi_query_rag import (
    MultiQueryRAG,
    SubQuery,
    FusedResult
)
from src.sota.cache_augmented_generation import (
    CacheAugmentedGeneration,
    CAGResponse,
    CAGContext,
    get_cag
)
from src.sota.multihop_graph_rag import (
    MultiHopGraphRAG,
    GraphEntity,
    GraphRelation,
    MultiHopResult,
    get_multihop_graph_rag
)
from src.sota.speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeResult,
    LookaheadDecoder,
    get_speculative_decoder
)

# Phase 4: Enterprise Scale
from src.sota.enterprise_features import (
    DistributedSearch,
    DistributedSearchResult,
    ABTestingFramework,
    Experiment,
    RateLimiter,
    get_distributed_search,
    get_ab_testing,
    get_rate_limiter
)

# Integration Layer
from src.sota.integration import (
    SOTAIntegration,
    SOTAQueryResult,
    get_sota_integration,
    initialize_sota
)


__all__ = [
    # Phase 1: Quick Wins
    "StreamingResponseGenerator",
    "StreamChunk",
    "SSEFormatter",
    "get_streamer",
    "AdaptiveRetriever",
    "QueryClassification",
    "QueryType",
    "TieredIntelligence",
    "IntelligenceTier",
    "TierDecision",
    "TierResponse",
    "get_tiered_intelligence",
    
    # Phase 2: Core Improvements
    "ColBERTReranker",
    "ColBERTResult",
    "get_colbert_reranker",
    "DomainAdaptedEmbedder",
    "EmbeddingResult",
    "TrainingPair",
    "get_domain_embedder",
    "EmbeddingWarmup",
    "WarmupStats",
    "get_embedding_warmup",
    "startup_warmup",
    
    # Phase 3: Advanced Features
    "MultiQueryRAG",
    "SubQuery",
    "FusedResult",
    "CacheAugmentedGeneration",
    "CAGResponse",
    "CAGContext",
    "get_cag",
    "MultiHopGraphRAG",
    "GraphEntity",
    "GraphRelation",
    "MultiHopResult",
    "get_multihop_graph_rag",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "LookaheadDecoder",
    "get_speculative_decoder",
    
    # Phase 4: Enterprise Scale
    "DistributedSearch",
    "DistributedSearchResult",
    "ABTestingFramework",
    "Experiment",
    "RateLimiter",
    "get_distributed_search",
    "get_ab_testing",
    "get_rate_limiter",
    
    # Integration Layer
    "SOTAIntegration",
    "SOTAQueryResult",
    "get_sota_integration",
    "initialize_sota",
    
    # Module functions
    "get_implementation_status",
    "get_sota_summary",
]


# Version and implementation status
VERSION = "1.0.0"
IMPLEMENTATION_STATUS = {
    "phase_1_quick_wins": {
        "response_streaming": "Complete",
        "adaptive_retrieval": "Complete",
        "tiered_intelligence": "Complete",
    },
    "phase_2_core_improvements": {
        "colbert_reranker": "Complete",
        "domain_embeddings": "Complete",
        "embedding_warmup": "Complete",
    },
    "phase_3_advanced": {
        "multi_query_rag": "Complete",
        "cache_augmented_generation": "Complete",
        "multihop_graph_rag": "Complete",
        "speculative_decoding": "Complete",
    },
    "phase_4_enterprise": {
        "distributed_search": "Complete",
        "ab_testing": "Complete",
        "rate_limiting": "Complete",
    }
}


def get_implementation_status() -> dict:
    """Get the implementation status of all SOTA features."""
    return IMPLEMENTATION_STATUS


def get_sota_summary() -> str:
    """Get a summary of implemented SOTA features."""
    total = sum(
        len(phase) for phase in IMPLEMENTATION_STATUS.values()
    )
    complete = sum(
        sum(1 for status in phase.values() if "Complete" in status)
        for phase in IMPLEMENTATION_STATUS.values()
    )
    
    return f"""SOTA RAG Module v{VERSION}
Implemented: {complete}/{total} features ({complete/total*100:.0f}%)

Expected Improvements:
- Cold start: 2-3s → <100ms (Embedding Warmup)
- LLM latency: 1-5s → <500ms for 70% queries (Tiered Intelligence)
- Recall: +15-20% (Multi-Query RAG)
- Reranking: 5-10x faster (ColBERT)
- Common queries: 40x faster (CAG)
"""
