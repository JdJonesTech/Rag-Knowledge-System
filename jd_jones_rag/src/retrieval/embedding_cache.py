"""
Production Embedding Cache
Redis-backed embedding cache for reduced API costs and faster retrieval.

Features:
- In-memory LRU cache for hot queries (using cachetools.TTLCache)
- Redis persistence for shared caching across instances
- Automatic TTL management
- Batch embedding support
- Statistics and monitoring

OPTIMIZATIONS:
- Uses cachetools.TTLCache for O(1) LRU operations instead of O(n) list-based eviction

SOTA INTEGRATION:
- Works with src.sota.embedding_warmup for cold start elimination
- Pre-warmed embeddings loaded at startup to eliminate 2-3s cold start
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import pickle
import numpy as np

# OPTIMIZATION: Use cachetools for efficient O(1) LRU operations
try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingCacheStats:
    """Statistics for embedding cache."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.memory_hits = 0
        self.redis_hits = 0
        self.api_calls_saved = 0
        self.estimated_cost_saved = 0.0  # In USD
    
    def record_hit(self, source: str = "memory"):
        self.hits += 1
        if source == "memory":
            self.memory_hits += 1
        elif source == "redis":
            self.redis_hits += 1
        self.api_calls_saved += 1
        # Estimate: $0.0001 per embedding (OpenAI ada-002)
        self.estimated_cost_saved += 0.0001
    
    def record_miss(self):
        self.misses += 1
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_hits": self.hits,
            "total_misses": self.misses,
            "memory_hits": self.memory_hits,
            "redis_hits": self.redis_hits,
            "hit_rate": f"{self.get_hit_rate():.2%}",
            "api_calls_saved": self.api_calls_saved,
            "estimated_cost_saved_usd": f"${self.estimated_cost_saved:.4f}"
        }


class ProductionEmbeddingCache:
    """
    Production-ready embedding cache with Redis backing.
    
    Architecture:
    1. L1 Cache: In-memory LRU (fastest, limited size)
    2. L2 Cache: Redis (shared across instances, larger capacity)
    3. Fallback: Generate new embedding from API
    """
    
    # Cost estimates per embedding (USD)
    EMBEDDING_COSTS = {
        "text-embedding-ada-002": 0.0001,
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
    }
    
    def __init__(
        self,
        redis_client=None,
        memory_max_size: int = 10000,
        redis_ttl_hours: int = 168,  # 1 week
        memory_ttl_seconds: int = 3600,  # 1 hour
        embedding_model: str = "text-embedding-ada-002",
        embedding_dim: int = 1536
    ):
        """
        Initialize production embedding cache.
        
        Args:
            redis_client: Async Redis client for L2 cache
            memory_max_size: Maximum L1 cache entries
            redis_ttl_hours: TTL for Redis entries
            memory_ttl_seconds: TTL for memory entries
            embedding_model: Model name for cost estimation
            embedding_dim: Expected embedding dimension
        """
        self.redis_client = redis_client
        self.memory_max_size = memory_max_size
        self.redis_ttl_hours = redis_ttl_hours
        self.memory_ttl_seconds = memory_ttl_seconds
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # OPTIMIZATION: L1 cache using cachetools.TTLCache for O(1) LRU operations
        # Falls back to manual dict if cachetools not available
        if CACHETOOLS_AVAILABLE:
            # TTLCache provides both LRU eviction AND automatic TTL expiry
            self.memory_cache = TTLCache(maxsize=memory_max_size, ttl=memory_ttl_seconds)
            self._use_cachetools = True
            logger.info(f"EmbeddingCache using cachetools.TTLCache (maxsize={memory_max_size}, ttl={memory_ttl_seconds}s)")
        else:
            # Fallback: manual dict-based cache
            self.memory_cache: Dict[str, Tuple[List[float], datetime]] = {}
            self._access_order: List[str] = []
            self._use_cachetools = False
            logger.warning("cachetools not available, using manual LRU cache")
        
        # Statistics
        self.stats = EmbeddingCacheStats()
        
        # Redis key prefix
        self.redis_prefix = "emb_cache:"
    
    def _compute_key(self, text: str) -> str:
        """Compute cache key for text."""
        # Normalize and hash
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _is_memory_entry_valid(self, key: str) -> bool:
        """Check if memory entry is still valid."""
        if self._use_cachetools:
            # OPTIMIZATION: cachetools.TTLCache handles expiry automatically
            return key in self.memory_cache
        else:
            # Fallback: manual TTL check
            if key not in self.memory_cache:
                return False
            _, timestamp = self.memory_cache[key]
            age_seconds = (datetime.now() - timestamp).total_seconds()
            return age_seconds < self.memory_ttl_seconds
    
    def _evict_memory_cache(self) -> None:
        """Evict old entries from memory cache."""
        if self._use_cachetools:
            # OPTIMIZATION: cachetools handles eviction automatically - O(1)
            pass
        else:
            # Fallback: manual eviction - O(n)
            while len(self.memory_cache) >= self.memory_max_size:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    self.memory_cache.pop(oldest_key, None)
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if self._use_cachetools:
            # OPTIMIZATION: cachetools updates access order automatically on get/set - O(1)
            pass
        else:
            # Fallback: manual access tracking - O(n) due to list.remove()
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def _get_from_memory(self, key: str) -> Optional[List[float]]:
        """Get embedding from memory cache."""
        if self._use_cachetools:
            # OPTIMIZATION: Direct O(1) access with automatic TTL
            return self.memory_cache.get(key)
        else:
            # Fallback: Manual tuple unpacking
            if key in self.memory_cache:
                embedding, _ = self.memory_cache[key]
                return embedding
            return None
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding (sync version for L1 only).
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None
        """
        key = self._compute_key(text)
        
        # Check L1 (memory) - OPTIMIZATION: O(1) with cachetools
        if self._is_memory_entry_valid(key):
            embedding = self._get_from_memory(key)
            if embedding is not None:
                self._update_access_order(key)
                self.stats.record_hit("memory")
                return embedding
        
        self.stats.record_miss()
        return None
    
    async def get_async(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding (async version, checks Redis).
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None
        """
        key = self._compute_key(text)
        
        # Check L1 (memory)
        if self._is_memory_entry_valid(key):
            embedding, _ = self.memory_cache[key]
            self._update_access_order(key)
            self.stats.record_hit("memory")
            return embedding
        
        # Check L2 (Redis)
        if self.redis_client:
            try:
                redis_key = f"{self.redis_prefix}{key}"
                cached_data = await self.redis_client.get(redis_key)
                
                if cached_data:
                    # Deserialize embedding
                    embedding = self._deserialize_embedding(cached_data)
                    
                    # Promote to L1
                    self._set_memory(key, embedding)
                    
                    self.stats.record_hit("redis")
                    return embedding
                    
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        
        self.stats.record_miss()
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding (sync, memory only).
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._compute_key(text)
        self._set_memory(key, embedding)
    
    async def set_async(
        self, 
        text: str, 
        embedding: List[float],
        persist_to_redis: bool = True
    ) -> None:
        """
        Cache an embedding (async, with Redis).
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            persist_to_redis: Whether to persist to Redis
        """
        key = self._compute_key(text)
        
        # Set in L1
        self._set_memory(key, embedding)
        
        # Set in L2 (Redis)
        if persist_to_redis and self.redis_client:
            try:
                redis_key = f"{self.redis_prefix}{key}"
                serialized = self._serialize_embedding(embedding)
                
                await self.redis_client.setex(
                    redis_key,
                    self.redis_ttl_hours * 3600,
                    serialized
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
    
    def _set_memory(self, key: str, embedding: List[float]) -> None:
        """Set embedding in memory cache."""
        self._evict_memory_cache()
        if self._use_cachetools:
            # OPTIMIZATION: cachetools stores value directly, handles TTL automatically
            self.memory_cache[key] = embedding
        else:
            # Fallback: store with timestamp for manual TTL
            self.memory_cache[key] = (embedding, datetime.now())
        self._update_access_order(key)
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding for Redis storage."""
        # Use numpy for efficient storage
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()
    
    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialize embedding from Redis."""
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.tolist()
    
    async def get_or_generate(
        self,
        text: str,
        embedding_generator
    ) -> List[float]:
        """
        Get cached embedding or generate new one.
        
        Args:
            text: Text to embed
            embedding_generator: Function/object to generate embeddings
            
        Returns:
            Embedding vector
        """
        # Try cache first
        cached = await self.get_async(text)
        if cached is not None:
            return cached
        
        # Generate new embedding
        try:
            if hasattr(embedding_generator, 'generate_embedding'):
                embedding = embedding_generator.generate_embedding(text)
            elif hasattr(embedding_generator, 'embed_query'):
                embedding = embedding_generator.embed_query(text)
            elif callable(embedding_generator):
                embedding = embedding_generator(text)
            else:
                raise ValueError("Invalid embedding generator")
            
            # Cache the new embedding
            await self.set_async(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    async def get_batch(
        self,
        texts: List[str]
    ) -> Tuple[Dict[str, List[float]], List[str]]:
        """
        Get cached embeddings for batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (cached_embeddings dict, uncached_texts list)
        """
        cached = {}
        uncached = []
        
        for text in texts:
            embedding = await self.get_async(text)
            if embedding is not None:
                cached[text] = embedding
            else:
                uncached.append(text)
        
        return cached, uncached
    
    async def set_batch(
        self,
        embeddings: Dict[str, List[float]]
    ) -> None:
        """
        Cache batch of embeddings.
        
        Args:
            embeddings: Dict of text -> embedding
        """
        for text, embedding in embeddings.items():
            await self.set_async(text, embedding)
    
    def invalidate(self, text: str) -> bool:
        """Invalidate a cached embedding."""
        key = self._compute_key(text)
        
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        
        return False
    
    async def invalidate_async(self, text: str) -> bool:
        """Invalidate cached embedding (including Redis)."""
        key = self._compute_key(text)
        removed = False
        
        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            removed = True
        
        # Remove from Redis
        if self.redis_client:
            try:
                redis_key = f"{self.redis_prefix}{key}"
                await self.redis_client.delete(redis_key)
                removed = True
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
        
        return removed
    
    def clear_memory(self) -> None:
        """Clear L1 memory cache."""
        self.memory_cache.clear()
        self._access_order.clear()
        logger.info("Memory cache cleared")
    
    async def clear_all(self) -> None:
        """Clear all caches (memory and Redis)."""
        self.clear_memory()
        
        if self.redis_client:
            try:
                # Delete all cache keys
                pattern = f"{self.redis_prefix}*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis cache entries")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats.update({
            "memory_cache_size": len(self.memory_cache),
            "memory_max_size": self.memory_max_size,
            "redis_enabled": self.redis_client is not None,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim
        })
        return stats
    
    async def warmup(
        self,
        common_queries: List[str],
        embedding_generator
    ) -> int:
        """
        Warmup cache with common queries.
        
        Args:
            common_queries: List of common queries to pre-cache
            embedding_generator: Embedding generator
            
        Returns:
            Number of embeddings generated and cached
        """
        count = 0
        for query in common_queries:
            try:
                # Check if already cached
                cached = await self.get_async(query)
                if cached is None:
                    # Generate and cache
                    await self.get_or_generate(query, embedding_generator)
                    count += 1
            except Exception as e:
                logger.warning(f"Warmup error for '{query[:30]}...': {e}")
        
        logger.info(f"Cache warmup complete: {count} new embeddings cached")
        return count


# Global instance
_embedding_cache = None

def get_embedding_cache(redis_client=None) -> ProductionEmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = ProductionEmbeddingCache(redis_client=redis_client)
    return _embedding_cache


# Integration helper for existing code
class CachedEmbeddingGenerator:
    """
    Wrapper that adds caching to any embedding generator.
    
    Usage:
        from src.data_ingestion.embedding_generator import EmbeddingGenerator
        
        base_generator = EmbeddingGenerator()
        cached_generator = CachedEmbeddingGenerator(base_generator)
        
        # Now embeddings are automatically cached
        embedding = cached_generator.generate_embedding("some text")
    """
    
    def __init__(
        self,
        base_generator,
        cache: Optional[ProductionEmbeddingCache] = None
    ):
        """
        Initialize cached embedding generator.
        
        Args:
            base_generator: The underlying embedding generator
            cache: Optional cache instance (uses global if not provided)
        """
        self.base_generator = base_generator
        self.cache = cache or get_embedding_cache()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching (sync)."""
        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Generate new
        embedding = self.base_generator.generate_embedding(text)
        
        # Cache it
        self.cache.set(text, embedding)
        
        return embedding
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding with caching (async)."""
        return await self.cache.get_or_generate(text, self.base_generator)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with caching."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate uncached embeddings
        if uncached_texts:
            new_embeddings = self.base_generator.generate_embeddings_batch(uncached_texts)
            
            # Fill in results and cache
            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                results[idx] = embedding
                self.cache.set(text, embedding)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
