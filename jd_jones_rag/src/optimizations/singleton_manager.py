"""
Singleton Manager
Manages singleton instances of expensive resources to avoid repeated initialization.

Optimizations:
- Single LLM client instance shared across all requests
- Pre-loaded embedding models with warm-up
- Connection pools for databases and Redis
- Thread-safe lazy initialization
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from threading import Lock
from functools import lru_cache
import atexit

logger = logging.getLogger(__name__)


class SingletonManager:
    """
    Thread-safe singleton manager for expensive resources.
    
    Features:
    - Lazy initialization with double-checked locking
    - Automatic cleanup on process exit
    - Health check methods for each resource
    - Connection pooling support
    """
    
    _instances: Dict[str, Any] = {}
    _locks: Dict[str, Lock] = {}
    _global_lock = Lock()
    
    @classmethod
    def _get_lock(cls, key: str) -> Lock:
        """Get or create a lock for a specific key."""
        if key not in cls._locks:
            with cls._global_lock:
                if key not in cls._locks:
                    cls._locks[key] = Lock()
        return cls._locks[key]
    
    @classmethod
    def get(cls, key: str, factory: callable, *args, **kwargs) -> Any:
        """
        Get or create a singleton instance.
        
        Args:
            key: Unique identifier for the instance
            factory: Callable to create the instance
            *args, **kwargs: Arguments for the factory
            
        Returns:
            The singleton instance
        """
        if key not in cls._instances:
            lock = cls._get_lock(key)
            with lock:
                if key not in cls._instances:
                    logger.info(f"Initializing singleton: {key}")
                    cls._instances[key] = factory(*args, **kwargs)
        return cls._instances[key]
    
    @classmethod
    def clear(cls, key: Optional[str] = None):
        """Clear singleton instance(s)."""
        if key:
            if key in cls._instances:
                instance = cls._instances.pop(key)
                if hasattr(instance, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(instance.close):
                            asyncio.create_task(instance.close())
                        else:
                            instance.close()
                    except Exception as e:
                        logger.error(f"Error closing {key}: {e}")
        else:
            for k in list(cls._instances.keys()):
                cls.clear(k)
    
    @classmethod
    def is_initialized(cls, key: str) -> bool:
        """Check if a singleton is initialized."""
        return key in cls._instances


# ============================================
# Pre-configured singleton factories
# ============================================

def _create_embedding_generator():
    """Create and warm up embedding generator."""
    from src.data_ingestion.embedding_generator import EmbeddingGenerator
    from src.config.settings import settings
    
    generator = EmbeddingGenerator(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=100
    )
    
    # Warm up the model with a test embedding
    try:
        warmup_text = "JD Jones industrial sealing solutions"
        generator.generate_embedding(warmup_text)
        logger.info("Embedding generator warmed up successfully")
    except Exception as e:
        logger.warning(f"Embedding warmup failed: {e}")
    
    return generator


def _create_llm_client():
    """Create LLM client with connection pooling."""
    from langchain_openai import ChatOpenAI
    from src.config.settings import settings
    
    provider = getattr(settings, 'llm_provider', 'openai').lower()
    
    if provider == 'ollama':
        ollama_base = getattr(settings, 'ollama_base_url', 'http://localhost:11434')
        client = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=f"{ollama_base}/v1",
            api_key="ollama",
            max_retries=3,
            request_timeout=30
        )
    else:
        client = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
            max_retries=3,
            request_timeout=30
        )
    
    logger.info(f"LLM client initialized with provider: {provider}")
    return client


def _create_redis_pool():
    """Create Redis connection pool."""
    import redis.asyncio as redis
    from src.config.settings import settings
    
    redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
    
    pool = redis.ConnectionPool.from_url(
        redis_url,
        max_connections=50,
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
        retry_on_timeout=True
    )
    
    logger.info("Redis connection pool created")
    return pool


def _create_db_pool():
    """Create async database connection pool."""
    import asyncpg
    from src.config.settings import settings
    
    # Note: This returns a coroutine, needs to be awaited
    async def create_pool():
        database_url = getattr(settings, 'database_url', '')
        # Extract connection parameters from URL
        if database_url.startswith('postgresql+asyncpg://'):
            database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
        
        pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=30,
            max_inactive_connection_lifetime=300
        )
        logger.info("Database connection pool created")
        return pool
    
    return create_pool


# ============================================
# Convenience functions
# ============================================

def get_embedding_generator():
    """Get the singleton embedding generator."""
    return SingletonManager.get('embedding_generator', _create_embedding_generator)


def get_llm_client():
    """Get the singleton LLM client."""
    return SingletonManager.get('llm_client', _create_llm_client)


def get_redis_pool():
    """Get the singleton Redis connection pool."""
    return SingletonManager.get('redis_pool', _create_redis_pool)


def get_db_pool():
    """Get the database pool factory."""
    return SingletonManager.get('db_pool_factory', _create_db_pool)


# Cleanup on exit
@atexit.register
def cleanup_singletons():
    """Clean up all singletons on process exit."""
    logger.info("Cleaning up singleton resources...")
    SingletonManager.clear()


# ============================================
# Cached helper functions (thread-safe)
# ============================================

@lru_cache(maxsize=1)
def get_sentence_transformer_model(model_name: str):
    """
    Get a cached SentenceTransformer model.
    Uses lru_cache for thread-safe caching.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        raise ImportError("sentence-transformers not installed")


@lru_cache(maxsize=1)
def get_cross_encoder_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Get a cached CrossEncoder model for reranking.
    """
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder model: {model_name}")
        model = CrossEncoder(model_name)
        return model
    except ImportError:
        raise ImportError("sentence-transformers not installed")
