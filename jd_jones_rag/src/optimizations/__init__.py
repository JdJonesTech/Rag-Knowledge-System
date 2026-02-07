"""
JD Jones RAG System - Performance Optimizations Module
Comprehensive optimizations for production-grade performance.

This module provides:
1. Singleton pattern for expensive resources
2. Connection pooling
3. Optimized caching with LRU eviction
4. Batch processing utilities
5. Memory management and garbage collection
6. Async utilities for parallelization
7. Startup optimizations
8. Inverted index for fast keyword search
"""

from src.optimizations.singleton_manager import (
    SingletonManager,
    get_embedding_generator,
    get_llm_client,
    get_redis_pool,
    get_db_pool
)

from src.optimizations.batch_processor import (
    BatchProcessor,
    BatchEmbeddingProcessor,
    BatchReranker
)

from src.optimizations.memory_optimizer import (
    MemoryOptimizer,
    session_cleanup_task,
    gc_optimizer,
    get_memory_optimizer
)

from src.optimizations.async_utils import (
    parallel_execute,
    with_timeout,
    gather_with_concurrency
)

from src.optimizations.cache_manager import (
    CacheManager,
    LRUCache,
    TTLCache,
    TwoLevelCache
)

from src.optimizations.inverted_index import (
    OptimizedBM25,
    InvertedIndex
)

from src.optimizations.startup import (
    StartupOptimizer,
    get_startup_optimizer,
    initialize_optimizations,
    shutdown_optimizations
)

from src.optimizations.optimized_reranker import (
    OptimizedReranker,
    RankedDocument,
    get_optimized_reranker
)

from src.optimizations.vector_index import (
    OptimizedVectorIndex,
    SearchResult,
    get_vector_index
)

__all__ = [
    # Singleton management
    'SingletonManager',
    'get_embedding_generator',
    'get_llm_client',
    'get_redis_pool',
    'get_db_pool',
    # Batch processing
    'BatchProcessor',
    'BatchEmbeddingProcessor',
    'BatchReranker',
    # Memory optimization
    'MemoryOptimizer',
    'session_cleanup_task',
    'gc_optimizer',
    'get_memory_optimizer',
    # Async utilities
    'parallel_execute',
    'with_timeout',
    'gather_with_concurrency',
    # Caching
    'CacheManager',
    'LRUCache',
    'TTLCache',
    'TwoLevelCache',
    # Search optimization
    'OptimizedBM25',
    'InvertedIndex',
    # Vector index (FAISS/Annoy/numpy)
    'OptimizedVectorIndex',
    'SearchResult',
    'get_vector_index',
    # Reranking
    'OptimizedReranker',
    'RankedDocument',
    'get_optimized_reranker',
    # Startup
    'StartupOptimizer',
    'get_startup_optimizer',
    'initialize_optimizations',
    'shutdown_optimizations'
]



