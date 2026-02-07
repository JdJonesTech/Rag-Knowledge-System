"""
Semantic Cache
Stores and reuses outputs for semantically similar queries.
Reduces latency and token costs for high-frequency queries.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached query-response pair."""
    query: str
    query_hash: str
    query_embedding: Optional[List[float]]
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_hash": self.query_hash,
            "response": self.response,
            "sources": self.sources,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count
        }


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    hits: int
    misses: int
    hit_rate: float
    avg_response_time_saved_ms: float
    memory_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "avg_response_time_saved_ms": self.avg_response_time_saved_ms,
            "memory_usage_mb": self.memory_usage_mb
        }


class SemanticCache:
    """
    Semantic cache for query-response pairs.
    
    Features:
    - Exact match caching (hash-based)
    - Semantic similarity caching (embedding-based)
    - TTL-based expiration
    - LRU eviction
    - Redis backend support (optional)
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_hours: int = 24,
        similarity_threshold: float = 0.92,
        embedding_generator=None,
        redis_client=None
    ):
        """
        Initialize semantic cache.
        
        Args:
            max_entries: Maximum cache entries
            default_ttl_hours: Default time-to-live in hours
            similarity_threshold: Minimum similarity for semantic match (0-1)
            embedding_generator: Embedding generator for semantic matching
            redis_client: Optional Redis client for distributed caching
        """
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours
        self.similarity_threshold = similarity_threshold
        self.embedding_generator = embedding_generator
        self.redis_client = redis_client
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.embedding_index: Dict[str, List[float]] = {}
        
        # OPTIMIZATION: Use OptimizedVectorIndex for O(log n) similarity search
        self._vector_index = None
        try:
            from src.optimizations.vector_index import OptimizedVectorIndex
            self._vector_index = OptimizedVectorIndex(dimension=384)
            logger.info("SemanticCache using OptimizedVectorIndex for similarity search")
        except ImportError:
            logger.debug("OptimizedVectorIndex not available, using linear search")
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, query: str) -> str:
        """Compute hash for a query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    async def get(
        self,
        query: str,
        use_semantic: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query.
        
        Args:
            query: Query string
            use_semantic: Whether to use semantic matching
            
        Returns:
            Cached response or None
        """
        # Try exact match first
        query_hash = self._compute_hash(query)
        
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if not entry.is_expired():
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                self.hits += 1
                return {
                    "response": entry.response,
                    "sources": entry.sources,
                    "cache_hit": True,
                    "match_type": "exact"
                }
        
        # Try semantic match
        if use_semantic and self.embedding_generator:
            try:
                query_embedding = await self._get_embedding(query)
                
                best_match = None
                best_similarity = 0
                
                # OPTIMIZATION: Use vector index for O(log n) search
                if self._vector_index and len(self._vector_index) > 0:
                    results = self._vector_index.search(query_embedding, top_k=1, min_score=self.similarity_threshold)
                    if results:
                        best_match = results[0].id
                        best_similarity = results[0].score
                else:
                    # Fallback: linear scan
                    for hash_key, embedding in self.embedding_index.items():
                        similarity = self._compute_similarity(query_embedding, embedding)
                        if similarity > best_similarity and similarity >= self.similarity_threshold:
                            best_similarity = similarity
                            best_match = hash_key
                
                if best_match and best_match in self.cache:
                    entry = self.cache[best_match]
                    if not entry.is_expired():
                        entry.hit_count += 1
                        entry.last_accessed = datetime.now()
                        self.hits += 1
                        return {
                            "response": entry.response,
                            "sources": entry.sources,
                            "cache_hit": True,
                            "match_type": "semantic",
                            "similarity": best_similarity
                        }
            except Exception as e:
                logger.error(f"Semantic cache lookup error: {e}")
        
        self.misses += 1
        return None
    
    async def set(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
        ttl_hours: int = None
    ) -> bool:
        """
        Cache a query-response pair.
        
        Args:
            query: Query string
            response: Response to cache
            sources: Source documents
            metadata: Additional metadata
            ttl_hours: Custom TTL in hours
            
        Returns:
            True if cached successfully
        """
        # Check cache size
        if len(self.cache) >= self.max_entries:
            self._evict_lru()
        
        query_hash = self._compute_hash(query)
        ttl = ttl_hours or self.default_ttl_hours
        
        # Get embedding for semantic matching
        query_embedding = None
        if self.embedding_generator:
            try:
                query_embedding = await self._get_embedding(query)
                self.embedding_index[query_hash] = query_embedding
                # OPTIMIZATION: Also add to vector index for ANN search
                if self._vector_index:
                    self._vector_index.add(query_hash, query_embedding)
            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
        
        entry = CacheEntry(
            query=query,
            query_hash=query_hash,
            query_embedding=query_embedding,
            response=response,
            sources=sources or [],
            metadata=metadata or {},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=ttl)
        )
        
        self.cache[query_hash] = entry
        
        # Also store in Redis if available
        if self.redis_client:
            try:
                await self._set_redis(query_hash, entry, ttl)
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")
        
        return True
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if hasattr(self.embedding_generator, 'embed_query'):
            return self.embedding_generator.embed_query(text)
        elif hasattr(self.embedding_generator, 'generate_embedding'):
            return await self.embedding_generator.generate_embedding(text)
        else:
            raise ValueError("Invalid embedding generator")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Sort by last accessed (oldest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        # Remove oldest 10%
        to_remove = max(1, len(sorted_entries) // 10)
        for i in range(to_remove):
            hash_key = sorted_entries[i][0]
            del self.cache[hash_key]
            if hash_key in self.embedding_index:
                del self.embedding_index[hash_key]
    
    def invalidate(self, query: str) -> bool:
        """
        Invalidate a cached entry.
        
        Args:
            query: Query to invalidate
            
        Returns:
            True if entry was found and removed
        """
        query_hash = self._compute_hash(query)
        if query_hash in self.cache:
            del self.cache[query_hash]
            if query_hash in self.embedding_index:
                del self.embedding_index[query_hash]
            return True
        return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching a pattern.
        
        Args:
            pattern: Pattern to match (substring)
            
        Returns:
            Number of entries invalidated
        """
        to_remove = []
        for hash_key, entry in self.cache.items():
            if pattern.lower() in entry.query.lower():
                to_remove.append(hash_key)
        
        for hash_key in to_remove:
            del self.cache[hash_key]
            if hash_key in self.embedding_index:
                del self.embedding_index[hash_key]
        
        return len(to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.embedding_index.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        # Estimate memory usage (rough)
        memory_bytes = sum(
            len(e.query) + len(e.response) + len(str(e.sources))
            for e in self.cache.values()
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        return CacheStats(
            total_entries=len(self.cache),
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            avg_response_time_saved_ms=500,  # Estimated
            memory_usage_mb=memory_mb
        )
    
    async def _set_redis(self, key: str, entry: CacheEntry, ttl_hours: int) -> None:
        """Store entry in Redis."""
        if not self.redis_client:
            return
        
        await self.redis_client.setex(
            f"cache:{key}",
            ttl_hours * 3600,
            json.dumps(entry.to_dict())
        )
    
    async def warmup(self, common_queries: List[str]) -> int:
        """
        Warmup cache with common queries.
        
        Args:
            common_queries: List of frequently asked queries
            
        Returns:
            Number of queries processed
        """
        # In production, this would pre-compute embeddings
        # for common queries to speed up semantic matching
        count = 0
        for query in common_queries:
            if self.embedding_generator:
                try:
                    embedding = await self._get_embedding(query)
                    query_hash = self._compute_hash(query)
                    self.embedding_index[query_hash] = embedding
                    count += 1
                except Exception:
                    pass
        return count
