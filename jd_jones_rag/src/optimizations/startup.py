"""
Application Startup Optimizations
Configure and initialize optimizations at application startup.

Integrates SOTA (State-of-the-Art) RAG enhancements:
- Embedding warmup for cold start elimination
- Tiered intelligence initialization
- Cache-Augmented Generation preloading
"""

import asyncio
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StartupOptimizer:
    """
    Handles all startup optimizations for the JD Jones RAG system.
    
    Call from main.py lifespan to initialize all optimizations.
    """
    
    def __init__(self):
        self._initialized = False
        self._memory_optimizer = None
        self._embedding_generator = None
        self._sota_integration = None
    
    async def initialize(self):
        """Initialize all optimizations."""
        if self._initialized:
            return
        
        logger.info("Initializing production optimizations...")
        
        # 1. Configure GC for optimal performance
        self._configure_gc()
        
        # 2. Initialize singleton resources
        await self._initialize_singletons()
        
        # 3. Start background tasks
        await self._start_background_tasks()
        
        # 4. Warm up models (legacy)
        await self._warmup_models()
        
        # 5. Initialize SOTA enhancements
        await self._initialize_sota()
        
        self._initialized = True
        logger.info("All optimizations initialized successfully")
    
    def _configure_gc(self):
        """Configure garbage collector for reduced latency."""
        # Increase thresholds to reduce GC frequency
        # This trades some memory for lower latency
        gen0, gen1, gen2 = gc.get_threshold()
        gc.set_threshold(gen0 * 2, gen1 * 2, gen2 * 2)
        gc.enable()
        logger.info(f"GC configured: thresholds = {gc.get_threshold()}")
    
    async def _initialize_singletons(self):
        """Initialize singleton resources."""
        try:
            from src.optimizations.singleton_manager import (
                get_embedding_generator,
                get_llm_client
            )
            
            # Initialize embedding generator (will warm up)
            self._embedding_generator = get_embedding_generator()
            logger.info("Embedding generator singleton initialized")
            
            # Initialize LLM client
            get_llm_client()
            logger.info("LLM client singleton initialized")
            
        except Exception as e:
            logger.warning(f"Some singletons failed to initialize: {e}")
    
    async def _start_background_tasks(self):
        """Start background optimization tasks."""
        try:
            from src.optimizations.memory_optimizer import get_memory_optimizer
            
            self._memory_optimizer = get_memory_optimizer()
            await self._memory_optimizer.start()
            logger.info("Memory optimizer started")
            
        except Exception as e:
            logger.warning(f"Background tasks failed to start: {e}")
    
    async def _warmup_models(self):
        """Warm up ML models with sample inputs."""
        try:
            if self._embedding_generator:
                # Already warmed up during initialization
                logger.info("Embedding model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def _initialize_sota(self):
        """Initialize SOTA (State-of-the-Art) RAG enhancements."""
        try:
            # 1. SOTA Embedding Warmup (eliminates 2-3s cold start)
            try:
                from src.sota.embedding_warmup import startup_warmup
                warmup_stats = await startup_warmup()
                logger.info(f"SOTA Embedding warmup: {warmup_stats.warmup_time_ms:.0f}ms, {len(warmup_stats.models_loaded)} models")
            except ImportError:
                logger.debug("SOTA embedding warmup not available")
            except Exception as e:
                logger.warning(f"SOTA embedding warmup failed: {e}")
            
            # 2. Initialize Tiered Intelligence (LLM → SLM → sklearn routing)
            try:
                from src.sota.tiered_intelligence import get_tiered_intelligence
                self._tiered_intelligence = get_tiered_intelligence()
                logger.info("SOTA Tiered Intelligence initialized")
            except ImportError:
                logger.debug("SOTA tiered intelligence not available")
            except Exception as e:
                logger.warning(f"SOTA tiered intelligence init failed: {e}")
            
            # 3. Initialize SOTA Integration Layer
            try:
                from src.sota.integration import get_sota_integration
                self._sota_integration = get_sota_integration()
                await self._sota_integration.initialize()
                logger.info("SOTA Integration layer initialized")
            except ImportError:
                logger.debug("SOTA integration not available")
            except Exception as e:
                logger.warning(f"SOTA integration init failed: {e}")
            
            logger.info("SOTA enhancements initialized")
            
        except Exception as e:
            logger.warning(f"SOTA initialization failed: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down optimizations...")
        
        if self._memory_optimizer:
            await self._memory_optimizer.stop()
        
        # Force final GC
        gc.collect()
        
        logger.info("Optimizations shutdown complete")


# Global instance
_startup_optimizer: Optional[StartupOptimizer] = None


def get_startup_optimizer() -> StartupOptimizer:
    """Get or create the startup optimizer."""
    global _startup_optimizer
    if _startup_optimizer is None:
        _startup_optimizer = StartupOptimizer()
    return _startup_optimizer


async def initialize_optimizations():
    """Initialize all optimizations. Call from app lifespan."""
    optimizer = get_startup_optimizer()
    await optimizer.initialize()


async def shutdown_optimizations():
    """Shutdown optimizations. Call from app lifespan."""
    optimizer = get_startup_optimizer()
    await optimizer.shutdown()
