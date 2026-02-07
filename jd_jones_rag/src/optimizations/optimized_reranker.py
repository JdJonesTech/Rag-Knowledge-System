"""
Optimized Reranker
Two-stage reranking with caching for improved latency.

OPTIMIZATIONS:
1. Two-stage: Fast filter (top 100) → Rerank (top 20)
2. Batch cross-encoder inference
3. Result caching with query+doc hash
4. Async processing
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """Document with reranking score."""
    content: str
    score: float
    original_score: float
    metadata: Dict[str, Any]
    doc_id: str


class OptimizedReranker:
    """
    Production-optimized reranker with two-stage approach.
    
    Stage 1: Fast filtering using lightweight scoring
    Stage 2: Deep reranking using cross-encoder on top candidates
    
    Features:
    - Two-stage pipeline for efficiency
    - Batch inference for cross-encoder
    - LRU cache for repeated queries
    - Async support
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        stage1_top_k: int = 100,
        stage2_top_k: int = 20,
        batch_size: int = 32,
        cache_size: int = 1000
    ):
        self.model_name = model_name
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.batch_size = batch_size
        self._model = None
        self._cache: Dict[str, float] = {}
        self._cache_size = cache_size
    
    def _get_model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using score passthrough")
        return self._model
    
    def _compute_cache_key(self, query: str, doc: str) -> str:
        """Compute cache key for query-doc pair."""
        combined = f"{query[:100]}||{doc[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _stage1_filter(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Fast filtering using original scores.
        
        OPTIMIZATION: Just sort by existing score, no model inference.
        """
        # Sort by original score and take top k
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        return sorted_docs[:top_k]
    
    def _stage2_rerank_batch(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """
        Stage 2: Deep reranking with cross-encoder in batches.
        
        OPTIMIZATION: Batch inference for efficiency.
        """
        model = self._get_model()
        if model is None:
            # Fallback: use original scores
            return [(i, doc.get("score", 0)) for i, doc in enumerate(documents)]
        
        # Check cache first
        pairs = []
        indices = []
        cached_scores = {}
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            cache_key = self._compute_cache_key(query, content)
            
            if cache_key in self._cache:
                cached_scores[i] = self._cache[cache_key]
            else:
                pairs.append((query, content))
                indices.append(i)
        
        # Batch inference for non-cached pairs
        if pairs:
            try:
                scores = model.predict(pairs, batch_size=self.batch_size)
                for idx, score in zip(indices, scores):
                    # Update cache
                    content = documents[idx].get("content", "")
                    cache_key = self._compute_cache_key(query, content)
                    self._cache[cache_key] = float(score)
                    cached_scores[idx] = float(score)
                    
                    # Evict old cache entries if needed
                    if len(self._cache) > self._cache_size:
                        # Remove first 10% of cache
                        keys_to_remove = list(self._cache.keys())[:self._cache_size // 10]
                        for k in keys_to_remove:
                            del self._cache[k]
            except Exception as e:
                logger.error(f"Cross-encoder inference failed: {e}")
                # Fallback to original scores
                for i in indices:
                    cached_scores[i] = documents[i].get("score", 0)
        
        # Combine all scores
        results = [(i, cached_scores.get(i, 0)) for i in range(len(documents))]
        return results
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Two-stage reranking pipeline.
        
        Args:
            query: Search query
            documents: List of documents with 'content' and optional 'score'
            top_k: Number of results to return (default: stage2_top_k)
            
        Returns:
            List of RankedDocument sorted by reranking score
        """
        top_k = top_k or self.stage2_top_k
        
        if not documents:
            return []
        
        # Stage 1: Fast filter to top candidates
        stage1_results = self._stage1_filter(query, documents, self.stage1_top_k)
        logger.debug(f"Stage 1: filtered to {len(stage1_results)} candidates")
        
        # Stage 2: Deep rerank with cross-encoder
        scored = self._stage2_rerank_batch(query, stage1_results)
        
        # Sort by reranking score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, score in scored[:top_k]:
            doc = stage1_results[idx]
            results.append(RankedDocument(
                content=doc.get("content", ""),
                score=score,
                original_score=doc.get("score", 0),
                metadata=doc.get("metadata", {}),
                doc_id=doc.get("document_id", doc.get("id", str(idx)))
            ))
        
        return results
    
    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Async version of rerank.
        Runs inference in thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_k
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_capacity": self._cache_size,
            "cache_hit_rate": "N/A"  # Would need to track hits/misses
        }


# Singleton instance
_reranker_instance: Optional[OptimizedReranker] = None


def get_optimized_reranker() -> OptimizedReranker:
    """Get singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = OptimizedReranker()
    return _reranker_instance
