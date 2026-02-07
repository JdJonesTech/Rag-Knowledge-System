"""
Memory Optimizer
Garbage collection and memory management optimizations.

Features:
1. Session cleanup with TTL
2. Automatic GC optimization
3. Memory pressure detection
4. Circular buffer for bounded memory
5. Weak references for cache entries
"""

import asyncio
import gc
import logging
import sys
import weakref
from typing import Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class SessionInfo:
    """Session metadata for cleanup tracking."""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    memory_usage_bytes: int = 0


class CircularBuffer(Generic[T]):
    """
    Fixed-size circular buffer for bounded memory usage.
    Automatically evicts oldest items when capacity is reached.
    """
    
    def __init__(self, max_size: int):
        """
        Initialize circular buffer.
        
        Args:
            max_size: Maximum number of items
        """
        self._buffer: deque[T] = deque(maxlen=max_size)
        self._lock = Lock()
    
    def append(self, item: T):
        """Add item, evicting oldest if at capacity."""
        with self._lock:
            self._buffer.append(item)
    
    def extend(self, items: List[T]):
        """Add multiple items."""
        with self._lock:
            self._buffer.extend(items)
    
    def get_all(self) -> List[T]:
        """Get all items as list."""
        with self._lock:
            return list(self._buffer)
    
    def get_latest(self, n: int) -> List[T]:
        """Get n most recent items."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]
    
    def clear(self):
        """Clear all items."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class WeakValueCache(Generic[T]):
    """
    Cache with weak references to values.
    Allows GC to collect items when memory is needed.
    """
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
        self._lock = Lock()
    
    def set(self, key: str, value: T):
        """Store value with weak reference."""
        with self._lock:
            self._cache[key] = weakref.ref(value)
    
    def get(self, key: str) -> Optional[T]:
        """Get value if still alive."""
        with self._lock:
            ref = self._cache.get(key)
            if ref is not None:
                value = ref()
                if value is not None:
                    return value
                # Clean up dead reference
                del self._cache[key]
            return None
    
    def cleanup(self):
        """Remove dead references."""
        with self._lock:
            dead_keys = [
                k for k, v in self._cache.items()
                if v() is None
            ]
            for k in dead_keys:
                del self._cache[k]
    
    def __len__(self) -> int:
        return len(self._cache)


class MemoryOptimizer:
    """
    Comprehensive memory optimization manager.
    
    Features:
    - Session lifecycle management
    - Automatic cleanup of stale resources
    - GC tuning for optimal performance
    - Memory pressure monitoring
    """
    
    # Memory thresholds (80% and 90% of available)
    WARNING_THRESHOLD = 0.80
    CRITICAL_THRESHOLD = 0.90
    
    def __init__(
        self,
        session_ttl_hours: int = 24,
        cleanup_interval_minutes: int = 15,
        max_sessions: int = 10000,
        max_conversation_messages: int = 50
    ):
        """
        Initialize memory optimizer.
        
        Args:
            session_ttl_hours: Session expiry time
            cleanup_interval_minutes: Cleanup check interval
            max_sessions: Maximum concurrent sessions
            max_conversation_messages: Max messages per conversation
        """
        self.session_ttl_hours = session_ttl_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.max_sessions = max_sessions
        self.max_conversation_messages = max_conversation_messages
        
        # Session tracking
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.stats = {
            'sessions_cleaned': 0,
            'gc_collections': 0,
            'memory_warnings': 0,
            'peak_memory_mb': 0
        }
        
        # Configure GC for optimal performance
        self._configure_gc()
    
    def _configure_gc(self):
        """Configure garbage collector for optimal performance."""
        # Get current thresholds
        gen0, gen1, gen2 = gc.get_threshold()
        
        # Increase thresholds to reduce GC frequency
        # This trades memory for lower latency
        gc.set_threshold(
            gen0 * 2,  # Gen0: More allocations before collection
            gen1 * 2,  # Gen1: More Gen0 collections before Gen1
            gen2 * 2   # Gen2: More Gen1 collections before Gen2
        )
        
        # Enable automatic garbage collection
        gc.enable()
        
        logger.info(f"GC configured: thresholds = {gc.get_threshold()}")
    
    async def start(self):
        """Start the memory optimizer."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Memory optimizer started")
    
    async def stop(self):
        """Stop the memory optimizer."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory optimizer stopped")
    
    async def register_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> SessionInfo:
        """Register a new session."""
        async with self._session_lock:
            now = datetime.now()
            session = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_activity=now
            )
            self._sessions[session_id] = session
            
            # Check if we need to evict old sessions
            if len(self._sessions) > self.max_sessions:
                await self._evict_oldest_sessions()
            
            return session
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity time."""
        async with self._session_lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_activity = datetime.now()
    
    async def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        async with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    async def _cleanup_loop(self):
        """Main cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _perform_cleanup(self):
        """Perform cleanup operations."""
        logger.info("Starting memory cleanup...")
        
        # 1. Clean up stale sessions
        stale_count = await self._cleanup_stale_sessions()
        
        # 2. Check memory pressure
        memory_info = self._get_memory_info()
        
        if memory_info['percent'] > self.CRITICAL_THRESHOLD:
            logger.warning(f"CRITICAL memory pressure: {memory_info['percent']:.1%}")
            self.stats['memory_warnings'] += 1
            # Aggressive cleanup
            self._force_gc_collection()
            await self._evict_oldest_sessions(count=len(self._sessions) // 4)
            
        elif memory_info['percent'] > self.WARNING_THRESHOLD:
            logger.warning(f"High memory pressure: {memory_info['percent']:.1%}")
            self.stats['memory_warnings'] += 1
            # Moderate cleanup
            self._force_gc_collection()
        
        # Update peak memory
        current_mb = memory_info['rss_mb']
        if current_mb > self.stats['peak_memory_mb']:
            self.stats['peak_memory_mb'] = current_mb
        
        logger.info(f"Cleanup complete: {stale_count} sessions removed, "
                   f"memory: {memory_info['rss_mb']:.1f}MB")
    
    async def _cleanup_stale_sessions(self) -> int:
        """Remove sessions past TTL."""
        cutoff = datetime.now() - timedelta(hours=self.session_ttl_hours)
        
        async with self._session_lock:
            stale = [
                sid for sid, info in self._sessions.items()
                if info.last_activity < cutoff
            ]
            
            for sid in stale:
                del self._sessions[sid]
            
            self.stats['sessions_cleaned'] += len(stale)
            return len(stale)
    
    async def _evict_oldest_sessions(self, count: Optional[int] = None):
        """Evict oldest sessions."""
        if count is None:
            count = max(1, len(self._sessions) // 10)  # 10% eviction
        
        async with self._session_lock:
            if not self._sessions:
                return
            
            # Sort by last activity
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            # Evict oldest
            for sid, _ in sorted_sessions[:count]:
                del self._sessions[sid]
            
            self.stats['sessions_cleaned'] += count
    
    def _force_gc_collection(self):
        """Force garbage collection."""
        collected = gc.collect()
        self.stats['gc_collections'] += 1
        logger.info(f"GC collected {collected} objects")
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage."""
        process = psutil.Process()
        memory = process.memory_info()
        
        return {
            'rss_mb': memory.rss / (1024 * 1024),
            'vms_mb': memory.vms / (1024 * 1024),
            'percent': process.memory_percent() / 100,
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        memory = self._get_memory_info()
        
        return {
            **self.stats,
            'active_sessions': len(self._sessions),
            'current_memory_mb': memory['rss_mb'],
            'memory_percent': memory['percent'],
            'gc_threshold': gc.get_threshold()
        }


# ============================================
# Global functions for convenience
# ============================================

_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get or create the global memory optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = MemoryOptimizer()
    return _optimizer


async def session_cleanup_task():
    """
    Background task for session cleanup.
    Should be started when the application starts.
    """
    optimizer = get_memory_optimizer()
    await optimizer.start()


def gc_optimizer():
    """
    Configure garbage collector for optimal performance.
    Call this during application startup.
    """
    optimizer = get_memory_optimizer()
    return optimizer


def create_conversation_buffer(max_messages: int = 50) -> CircularBuffer:
    """
    Create a circular buffer for conversation history.
    
    Args:
        max_messages: Maximum messages to keep
        
    Returns:
        CircularBuffer instance
    """
    return CircularBuffer(max_messages)
