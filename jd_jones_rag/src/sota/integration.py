"""
SOTA RAG Integration Layer

Integrates all SOTA features with the existing JD Jones RAG system.
Provides a unified interface for using the optimized pipeline.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SOTAQueryResult:
    """Result from the SOTA-optimized query pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    tier_used: str
    latency_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    from_cache: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources[:5],
            "tier_used": self.tier_used,
            "latency_ms": round(self.latency_ms, 2),
            "from_cache": self.from_cache,
            "confidence": round(self.confidence, 2)
        }


class SOTAIntegration:
    """
    SOTA RAG Integration Layer.
    
    Provides a unified, optimized query interface that:
    1. Uses tiered intelligence for fast routing
    2. Applies adaptive retrieval
    3. Uses ColBERT reranking
    4. Leverages CAG for common queries
    5. Streams responses when appropriate
    
    Usage:
        sota = SOTAIntegration()
        await sota.initialize()
        result = await sota.query("What is NA 701?")
    """
    
    def __init__(
        self,
        enable_streaming: bool = True,
        enable_cag: bool = True,
        enable_tiered: bool = True,
        enable_colbert: bool = True
    ):
        """
        Initialize SOTA integration.
        
        Args:
            enable_streaming: Enable response streaming
            enable_cag: Enable Cache-Augmented Generation
            enable_tiered: Enable tiered intelligence
            enable_colbert: Enable ColBERT reranking
        """
        self.enable_streaming = enable_streaming
        self.enable_cag = enable_cag
        self.enable_tiered = enable_tiered
        self.enable_colbert = enable_colbert
        
        self._initialized = False
        self._tiered = None
        self._cag = None
        self._colbert = None
        self._adaptive = None
        
        self._stats = {
            "queries_processed": 0,
            "avg_latency_ms": 0,
            "cache_hits": 0,
            "tier_1_count": 0,
            "tier_2_count": 0,
            "tier_3_count": 0
        }
    
    async def initialize(self):
        """Initialize all SOTA components."""
        if self._initialized:
            return
        
        logger.info("Initializing SOTA integration...")
        start = time.time()
        
        # Run warmup
        try:
            from src.sota.embedding_warmup import startup_warmup
            await startup_warmup()
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        # Initialize tiered intelligence
        if self.enable_tiered:
            try:
                from src.sota.tiered_intelligence import get_tiered_intelligence
                self._tiered = get_tiered_intelligence()
            except Exception as e:
                logger.warning(f"Tiered intelligence init failed: {e}")
        
        # Initialize CAG
        if self.enable_cag:
            try:
                from src.sota.cache_augmented_generation import get_cag
                self._cag = get_cag()
            except Exception as e:
                logger.warning(f"CAG init failed: {e}")
        
        # Initialize ColBERT
        if self.enable_colbert:
            try:
                from src.sota.colbert_reranker import get_colbert_reranker
                self._colbert = get_colbert_reranker()
            except Exception as e:
                logger.warning(f"ColBERT init failed: {e}")
        
        # Initialize adaptive retrieval
        try:
            from src.sota.adaptive_retrieval import AdaptiveRetriever
            self._adaptive = AdaptiveRetriever()
        except Exception as e:
            logger.warning(f"Adaptive retrieval init failed: {e}")
        
        self._initialized = True
        logger.info(f"SOTA initialization complete in {(time.time()-start)*1000:.0f}ms")
    
    async def query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> SOTAQueryResult:
        """
        Process a query through the SOTA pipeline.
        
        Args:
            query: User query
            context: Optional context
            stream: Enable streaming response
            
        Returns:
            SOTAQueryResult with optimized response
        """
        if not self._initialized:
            await self.initialize()
        
        start = time.time()
        context = context or {}
        
        self._stats["queries_processed"] += 1
        
        # Try tiered intelligence first
        if self._tiered and self.enable_tiered:
            response = await self._tiered.process(query, context)
            
            tier_name = response.tier_used.name
            if "SKLEARN" in tier_name:
                self._stats["tier_1_count"] += 1
            elif "SLM" in tier_name:
                self._stats["tier_2_count"] += 1
            else:
                self._stats["tier_3_count"] += 1
            
            if response.content and response.confidence > 0.7:
                latency = (time.time() - start) * 1000
                self._update_avg_latency(latency)
                
                return SOTAQueryResult(
                    answer=response.content,
                    sources=[],
                    tier_used=tier_name,
                    latency_ms=latency,
                    retrieval_time_ms=0,
                    generation_time_ms=response.latency_ms,
                    from_cache=False,
                    confidence=response.confidence
                )
        
        # Try CAG for potential cache hit
        if self._cag and self.enable_cag:
            cag_response = await self._cag.generate(query, context.get("retrieved_context"))
            
            if cag_response.used_cache:
                self._stats["cache_hits"] += 1
                latency = (time.time() - start) * 1000
                self._update_avg_latency(latency)
                
                return SOTAQueryResult(
                    answer=cag_response.answer,
                    sources=[],
                    tier_used="CAG",
                    latency_ms=latency,
                    retrieval_time_ms=cag_response.retrieval_time_ms,
                    generation_time_ms=cag_response.generation_time_ms,
                    from_cache=True,
                    confidence=cag_response.confidence
                )
        
        # Full pipeline with adaptive retrieval
        retrieval_start = time.time()
        retrieval_result = None
        sources = []
        
        if self._adaptive:
            retrieval_result = await self._adaptive.retrieve(query)
            sources = retrieval_result.get("results", [])
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Rerank with ColBERT if available
        if self._colbert and sources and self.enable_colbert:
            reranked = self._colbert.rerank(query, sources, top_k=5)
            sources = [r.to_dict() for r in reranked]
        
        # Generate response
        generation_start = time.time()
        
        if self._cag:
            context_str = "\n".join([
                src.get("content", "")[:500]
                for src in sources[:3]
            ])
            cag_response = await self._cag.generate(query, context_str)
            answer = cag_response.answer
            confidence = cag_response.confidence
        else:
            answer = "Unable to generate response"
            confidence = 0
        
        generation_time = (time.time() - generation_start) * 1000
        latency = (time.time() - start) * 1000
        self._update_avg_latency(latency)
        
        return SOTAQueryResult(
            answer=answer,
            sources=sources,
            tier_used="FULL_PIPELINE",
            latency_ms=latency,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            from_cache=False,
            confidence=confidence
        )
    
    def _update_avg_latency(self, latency: float):
        """Update running average latency."""
        n = self._stats["queries_processed"]
        current_avg = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = ((n - 1) * current_avg + latency) / n
    
    async def stream_query(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        Stream query response.
        
        Yields:
            StreamChunk objects
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            from src.sota.response_streaming import get_streamer
            streamer = get_streamer()
            
            # Get context from retrieval
            context_str = ""
            if self._adaptive:
                result = await self._adaptive.retrieve(query, top_k=3)
                sources = result.get("results", [])
                context_str = "\n".join([
                    src.get("content", "")[:500]
                    for src in sources[:3]
                ])
            
            async for chunk in streamer.stream(query, context_str):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            from src.sota.response_streaming import StreamChunk
            yield StreamChunk(
                content=f"Error: {str(e)}",
                chunk_index=0,
                is_final=True
            )
    
    def preload_products(self, products: List[Dict[str, Any]]) -> int:
        """Preload products for CAG."""
        if self._cag:
            return self._cag.preload_products(products)
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = {
            **self._stats,
            "avg_latency_ms": round(self._stats["avg_latency_ms"], 2),
            "initialized": self._initialized,
            "components": {
                "tiered": self._tiered is not None,
                "cag": self._cag is not None,
                "colbert": self._colbert is not None,
                "adaptive": self._adaptive is not None
            }
        }
        
        # Add component stats
        if self._tiered:
            stats["tiered_stats"] = self._tiered.get_stats()
        if self._cag:
            stats["cag_stats"] = self._cag.get_stats()
        if self._colbert:
            stats["colbert_stats"] = self._colbert.get_stats()
        
        return stats


# Singleton instance
_sota_integration: Optional[SOTAIntegration] = None


def get_sota_integration() -> SOTAIntegration:
    """Get singleton SOTA integration instance."""
    global _sota_integration
    if _sota_integration is None:
        _sota_integration = SOTAIntegration()
    return _sota_integration


async def initialize_sota():
    """Initialize SOTA system at application startup."""
    integration = get_sota_integration()
    await integration.initialize()
    return integration
