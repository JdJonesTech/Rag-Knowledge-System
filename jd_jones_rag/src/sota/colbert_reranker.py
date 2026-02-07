"""
ColBERTv2 Late Interaction Reranker

Implements the ColBERT (Contextualized Late Interaction over BERT) approach for 
high-precision reranking with token-level matching.

PRECISION vs SPEED TRADE-OFF:
- ColBERT uses per-token embeddings (not dense single-vector)
- HIGHER PRECISION than dense vectors due to late interaction
- 5-10x FASTER than cross-encoders (which process full query+doc pairs)
- SLOWER than dense vector similarity (trade-off for precision)

When to use ColBERT:
- When precision matters more than raw speed
- For technical/domain-specific vocabulary matching
- When reranking a small candidate set (10-100 docs)

Dependencies:
- RAGatouille (wraps ColBERT): pip install ragatouille
- OR: sentence-transformers for CrossEncoder fallback
- Underlying: transformers, torch (BERT-based models)

Reference:
- ColBERTv2: https://arxiv.org/abs/2112.01488
- RAGatouille library for easy ColBERT integration
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies - ColBERT/BERT models
try:
    from ragatouille import RAGPretrainedModel
    RAGATOUILLE_AVAILABLE = True
except ImportError:
    RAGATOUILLE_AVAILABLE = False
    logger.info("RAGatouille not installed - ColBERT will use fallback mode")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# Note: Both ragatouille and sentence_transformers internally use:
# - transformers (HuggingFace) for BERT models
# - torch for tensor operations
# These are installed as dependencies of the above packages


@dataclass
class ColBERTResult:
    """Result from ColBERT reranking."""
    doc_id: str
    content: str
    score: float
    original_score: float
    rank: int
    token_matches: List[Tuple[str, str, float]] = field(default_factory=list)  # (query_token, doc_token, score)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "score": self.score,
            "original_score": self.original_score,
            "rank": self.rank,
            "token_matches": self.token_matches[:5],  # Top 5 matches
            "metadata": self.metadata
        }


class ColBERTReranker:
    """
    ColBERTv2 Late Interaction Reranker.
    
    Uses late interaction scoring for efficient, high-precision reranking:
    1. Encode query and document tokens separately
    2. Compute MaxSim scores between token embeddings
    3. Sum across query tokens for final relevance score
    
    Benefits over Cross-Encoder:
    - 5-10x faster at inference time
    - Better token-level matching for technical terms
    - Supports pre-indexing for sub-millisecond reranking
    
    Usage:
        reranker = ColBERTReranker()
        results = await reranker.rerank(query, documents, top_k=10)
    """
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_path: Optional[str] = None,
        use_gpu: bool = True,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """
        Initialize ColBERT reranker.
        
        Args:
            model_name: ColBERT model name (from HuggingFace)
            fallback_model: Cross-encoder model to use if ColBERT unavailable
            index_path: Path to pre-built ColBERT index (optional)
            use_gpu: Whether to use GPU for inference
            cache_size: Size of score cache
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.cache_size = cache_size
        self.batch_size = batch_size
        
        self._colbert_model = None
        self._fallback_encoder = None
        self._cache: Dict[str, float] = {}
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "rerank_calls": 0,
            "colbert_used": 0,
            "fallback_used": 0
        }
    
    def _get_colbert_model(self):
        """Lazy load ColBERT model."""
        if self._colbert_model is None and RAGATOUILLE_AVAILABLE:
            try:
                self._colbert_model = RAGPretrainedModel.from_pretrained(
                    self.model_name,
                    index_name="jd_jones_colbert"
                )
                logger.info(f"Loaded ColBERT model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load ColBERT model: {e}")
        return self._colbert_model
    
    def _get_fallback_encoder(self):
        """Lazy load fallback cross-encoder."""
        if self._fallback_encoder is None and CROSS_ENCODER_AVAILABLE:
            try:
                self._fallback_encoder = CrossEncoder(self.fallback_model)
                logger.info(f"Loaded fallback cross-encoder: {self.fallback_model}")
            except Exception as e:
                logger.warning(f"Failed to load fallback encoder: {e}")
        return self._fallback_encoder
    
    def _compute_cache_key(self, query: str, doc: str) -> str:
        """Compute cache key for query-doc pair."""
        combined = f"{query[:100]}||{doc[:300]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _maxsim_score(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Compute MaxSim score (core of ColBERT late interaction).
        
        For each query token, find the maximum similarity to any document token,
        then sum across all query tokens.
        
        Args:
            query_embeddings: Shape (num_query_tokens, embedding_dim)
            doc_embeddings: Shape (num_doc_tokens, embedding_dim)
            
        Returns:
            Tuple of (total_score, list of (query_idx, doc_idx, score) matches)
        """
        # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        # For each query token, get max similarity to any doc token
        max_sims = np.max(similarity_matrix, axis=1)  # (num_query_tokens,)
        max_indices = np.argmax(similarity_matrix, axis=1)  # Best doc token for each query token
        
        # Collect top matches
        matches = []
        for q_idx, (d_idx, score) in enumerate(zip(max_indices, max_sims)):
            matches.append((q_idx, int(d_idx), float(score)))
        
        # Sum of max similarities
        total_score = float(np.sum(max_sims))
        
        return total_score, matches
    
    def _colbert_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[ColBERTResult]:
        """
        Rerank using ColBERT model.
        
        Uses RAGatouille for ColBERT inference.
        """
        model = self._get_colbert_model()
        if model is None:
            return []
        
        self._stats["colbert_used"] += 1
        
        try:
            # Extract document content
            doc_contents = [doc.get("content", "") for doc in documents]
            
            # ColBERT rerank
            colbert_results = model.rerank(
                query=query,
                documents=doc_contents,
                k=top_k
            )
            
            # Build results
            results = []
            for rank, (doc_idx, score) in enumerate(colbert_results):
                doc = documents[doc_idx]
                results.append(ColBERTResult(
                    doc_id=doc.get("document_id", doc.get("id", str(doc_idx))),
                    content=doc.get("content", ""),
                    score=score,
                    original_score=doc.get("score", 0),
                    rank=rank,
                    metadata=doc.get("metadata", {})
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"ColBERT rerank failed: {e}")
            return []
    
    def _fallback_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[ColBERTResult]:
        """
        Fallback reranking using cross-encoder.
        """
        encoder = self._get_fallback_encoder()
        self._stats["fallback_used"] += 1
        
        if encoder is None:
            # Last resort: sort by original score
            sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
            return [
                ColBERTResult(
                    doc_id=doc.get("document_id", doc.get("id", str(i))),
                    content=doc.get("content", ""),
                    score=doc.get("score", 0),
                    original_score=doc.get("score", 0),
                    rank=i,
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(sorted_docs[:top_k])
            ]
        
        try:
            # Create query-document pairs
            pairs = [(query, doc.get("content", "")) for doc in documents]
            
            # Check cache
            scores = []
            to_compute = []
            to_compute_indices = []
            
            for i, (q, d) in enumerate(pairs):
                cache_key = self._compute_cache_key(q, d)
                if cache_key in self._cache:
                    scores.append((i, self._cache[cache_key]))
                    self._stats["cache_hits"] += 1
                else:
                    to_compute.append((q, d))
                    to_compute_indices.append(i)
                    self._stats["cache_misses"] += 1
            
            # Batch inference for non-cached pairs
            if to_compute:
                computed_scores = encoder.predict(to_compute, batch_size=self.batch_size)
                for idx, score in zip(to_compute_indices, computed_scores):
                    cache_key = self._compute_cache_key(pairs[idx][0], pairs[idx][1])
                    self._cache[cache_key] = float(score)
                    scores.append((idx, float(score)))
                    
                    # Cache eviction
                    if len(self._cache) > self.cache_size:
                        keys_to_remove = list(self._cache.keys())[:self.cache_size // 10]
                        for k in keys_to_remove:
                            del self._cache[k]
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Build results
            results = []
            for rank, (idx, score) in enumerate(scores[:top_k]):
                doc = documents[idx]
                results.append(ColBERTResult(
                    doc_id=doc.get("document_id", doc.get("id", str(idx))),
                    content=doc.get("content", ""),
                    score=score,
                    original_score=doc.get("score", 0),
                    rank=rank,
                    metadata=doc.get("metadata", {})
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback rerank failed: {e}")
            return []
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[ColBERTResult]:
        """
        Rerank documents using ColBERT late interaction.
        
        Args:
            query: Search query
            documents: List of documents with 'content' field
            top_k: Number of results to return
            
        Returns:
            List of ColBERTResult sorted by relevance
        """
        self._stats["rerank_calls"] += 1
        
        if not documents:
            return []
        
        # Try ColBERT first
        if RAGATOUILLE_AVAILABLE:
            results = self._colbert_rerank(query, documents, top_k)
            if results:
                return results
        
        # Fallback to cross-encoder
        return self._fallback_rerank(query, documents, top_k)
    
    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[ColBERTResult]:
        """
        Async version of rerank.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_k
        )
    
    def build_index(
        self,
        documents: List[Dict[str, Any]],
        index_path: str
    ) -> bool:
        """
        Build ColBERT index for a document collection.
        
        Pre-computing document embeddings enables sub-millisecond reranking.
        
        Args:
            documents: Documents to index
            index_path: Path to save the index
            
        Returns:
            True if successful
        """
        model = self._get_colbert_model()
        if model is None:
            logger.error("ColBERT model not available for indexing")
            return False
        
        try:
            doc_contents = [doc.get("content", "") for doc in documents]
            doc_ids = [doc.get("document_id", doc.get("id", str(i))) for i, doc in enumerate(documents)]
            
            # Build index
            model.index(
                documents=doc_contents,
                document_ids=doc_ids,
                index_name="jd_jones_products",
                split_documents=True
            )
            
            logger.info(f"Built ColBERT index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build ColBERT index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "colbert_available": RAGATOUILLE_AVAILABLE,
            "fallback_available": CROSS_ENCODER_AVAILABLE
        }


# Singleton instance
_colbert_reranker: Optional[ColBERTReranker] = None


def get_colbert_reranker() -> ColBERTReranker:
    """Get singleton ColBERT reranker instance."""
    global _colbert_reranker
    if _colbert_reranker is None:
        _colbert_reranker = ColBERTReranker()
    return _colbert_reranker
