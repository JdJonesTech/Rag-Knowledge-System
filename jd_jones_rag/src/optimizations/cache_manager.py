"""
Cache Manager - Optimized LRU, TTL, and Two-Level Caching
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    value: T
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at
    
    def touch(self):
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache(Generic[T]):
    """LRU Cache with O(1) operations using OrderedDict."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = Lock()
        self._current_size_bytes = 0
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get(self, key: str) -> Optional[T]:
        with self._lock:
            if key not in self._cache:
                self.stats['misses'] += 1
                return None
            entry = self._cache[key]
            if entry.is_expired():
                self._remove(key)
                self.stats['misses'] += 1
                return None
            self._cache.move_to_end(key)
            entry.touch()
            self.stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: T, ttl_seconds: Optional[float] = None):
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 1000
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds else None
        entry = CacheEntry(value=value, created_at=now, expires_at=expires_at, size_bytes=size_bytes)
        with self._lock:
            if key in self._cache:
                self._remove(key)
            self._evict_if_needed(size_bytes)
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
    
    def _remove(self, key: str):
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size_bytes -= entry.size_bytes
    
    def _evict_if_needed(self, new_size: int):
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self.stats['evictions'] += 1
        if self.max_memory_bytes:
            while self._current_size_bytes + new_size > self.max_memory_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
                self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['hits'] + self.stats['misses']
        return {**self.stats, 'size': len(self._cache), 'max_size': self.max_size,
                'memory_bytes': self._current_size_bytes, 'hit_rate': self.stats['hits'] / total if total > 0 else 0}
    
    def __len__(self): return len(self._cache)
    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired()


class TTLCache(LRUCache[T]):
    """LRU Cache with default TTL and background cleanup."""
    
    def __init__(self, max_size: int = 1000, default_ttl_seconds: float = 3600):
        super().__init__(max_size)
        self.default_ttl = default_ttl_seconds
    
    def set(self, key: str, value: T, ttl_seconds: Optional[float] = None):
        super().set(key, value, ttl_seconds or self.default_ttl)


class TwoLevelCache(Generic[T]):
    """Two-level cache: L1 (memory) + L2 (Redis)."""
    
    def __init__(self, l1_max_size: int = 1000, l1_ttl_seconds: float = 300,
                 l2_ttl_seconds: float = 3600, redis_client=None, key_prefix: str = "cache:"):
        self.l1 = TTLCache(max_size=l1_max_size, default_ttl_seconds=l1_ttl_seconds)
        self.l2_ttl = l2_ttl_seconds
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}
    
    async def get(self, key: str) -> Optional[T]:
        value = self.l1.get(key)
        if value is not None:
            self.stats['l1_hits'] += 1
            return value
        if self.redis:
            try:
                data = await self.redis.get(f"{self.key_prefix}{key}")
                if data:
                    value = pickle.loads(data)
                    self.l1.set(key, value)
                    self.stats['l2_hits'] += 1
                    return value
            except Exception as e:
                logger.debug(f"L2 get error: {e}")
        self.stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: T, l1_only: bool = False):
        self.l1.set(key, value)
        if not l1_only and self.redis:
            try:
                await self.redis.setex(f"{self.key_prefix}{key}", int(self.l2_ttl), pickle.dumps(value))
            except Exception as e:
                logger.debug(f"L2 set error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['misses']
        return {**self.stats, 'l1_stats': self.l1.get_stats(),
                'hit_rate': (self.stats['l1_hits'] + self.stats['l2_hits']) / total if total > 0 else 0}


class CacheManager:
    """Central cache management with named instances."""
    _caches: Dict[str, Union[LRUCache, TTLCache, TwoLevelCache]] = {}
    _lock = Lock()
    
    @classmethod
    def get_cache(cls, name: str, cache_type: str = "lru", **kwargs):
        with cls._lock:
            if name not in cls._caches:
                if cache_type == "lru":
                    cls._caches[name] = LRUCache(**kwargs)
                elif cache_type == "ttl":
                    cls._caches[name] = TTLCache(**kwargs)
                elif cache_type == "two_level":
                    cls._caches[name] = TwoLevelCache(**kwargs)
            return cls._caches[name]
    
    @classmethod
    def clear_all(cls):
        with cls._lock:
            for cache in cls._caches.values():
                cache.clear()
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        with cls._lock:
            return {name: cache.get_stats() for name, cache in cls._caches.items()}


def cached(cache_name: str = "default", ttl_seconds: Optional[float] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = CacheManager.get_cache(cache_name, "ttl", max_size=1000)
            key = hashlib.md5(json.dumps([args, kwargs], default=str).encode()).hexdigest()
            result = cache.get(key)
            if result is not None:
                return result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = CacheManager.get_cache(cache_name, "ttl", max_size=1000)
            key = hashlib.md5(json.dumps([args, kwargs], default=str).encode()).hexdigest()
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
