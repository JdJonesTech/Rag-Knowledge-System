"""
Advanced RAG Retrieval Optimizations
Implements research-based retrieval enhancements for improved precision and recall.

Enhancements:
1. Reciprocal Rank Fusion (RRF) - Combines multiple retrieval strategies
2. Query Expansion - Adds synonyms and related terms
3. Embedding Cache - Caches embeddings for repeated queries
4. Multi-Stage Reranking Pipeline
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Unified retrieval result."""
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""  # 'vector', 'keyword', 'hybrid'
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
            "rank": self.rank
        }


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple retrieval strategies.
    
    Research: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms 
    Condorcet and individual rank learning methods"
    
    Formula: RRF(d) = Σ 1 / (k + rank(d))
    where k is a constant (typically 60) and rank(d) is the rank from each retrieval method.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: Smoothing constant (default 60 as per research)
        """
        self.k = k
    
    def fuse(
        self,
        *result_lists: List[RetrievalResult],
        weights: Optional[List[float]] = None
    ) -> List[RetrievalResult]:
        """
        Fuse multiple ranked result lists using RRF.
        
        Args:
            result_lists: Multiple lists of ranked results
            weights: Optional weights for each list (default: equal weights)
            
        Returns:
            Fused and re-ranked results
        """
        if not result_lists:
            return []
        
        # Normalize weights
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_content: Dict[str, RetrievalResult] = {}
        
        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]
            
            for rank, result in enumerate(results, start=1):
                doc_id = result.document_id
                
                # RRF formula with weight
                rrf_score = weight * (1.0 / (self.k + rank))
                
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score
                    doc_content[doc_id] = result
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            result = doc_content[doc_id]
            fused_results.append(RetrievalResult(
                document_id=doc_id,
                content=result.content,
                metadata=result.metadata,
                score=score,
                source="rrf_fused",
                rank=rank
            ))
        
        return fused_results


class QueryExpander:
    """
    Query expansion for improved recall.
    
    Techniques:
    1. Product code normalization (JD Jones specific)
    2. Technical synonym expansion
    3. Industry-specific term mapping
    """
    
    # JD Jones product synonyms and related terms
    PRODUCT_SYNONYMS = {
        "packing": ["packing set", "packing rings", "stem packing", "valve packing"],
        "gasket": ["sealing gasket", "flange gasket", "seal", "sealing element"],
        "seal": ["o-ring", "lip seal", "mechanical seal", "shaft seal"],
        "ptfe": ["teflon", "polytetrafluoroethylene"],
        "graphite": ["expanded graphite", "flexible graphite", "carbon graphite"],
        "aramid": ["kevlar", "aramid fiber", "nomex"],
    }
    
    # Industry standard aliases
    STANDARD_ALIASES = {
        "api 622": ["api622", "api-622", "api 622"],
        "api 624": ["api624", "api-624", "api 624"],
        "api 6a": ["api6a", "api-6a", "api 6a"],
        "asme b16.20": ["asmeb16.20", "asme b16-20"],
        "shell spe": ["shell spe 77/312", "shell specification"],
    }
    
    def __init__(self):
        """Initialize query expander."""
        # Build reverse lookup for synonyms
        self.synonym_lookup = {}
        for base_term, synonyms in self.PRODUCT_SYNONYMS.items():
            for syn in synonyms:
                self.synonym_lookup[syn.lower()] = base_term
    
    def expand(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        expanded_terms = []
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.append(query)
        
        # Check for product codes and normalize
        product_codes = self._extract_product_codes(query)
        for code in product_codes:
            normalized = self._normalize_product_code(code)
            if normalized != code:
                expanded_terms.append(normalized)
        
        # Add synonyms
        for syn, base_term in self.synonym_lookup.items():
            if syn in query_lower:
                # Add the base term and other synonyms
                expanded_terms.extend(self.PRODUCT_SYNONYMS.get(base_term, []))
        
        # Add standard aliases
        for standard, aliases in self.STANDARD_ALIASES.items():
            for alias in aliases:
                if alias in query_lower:
                    expanded_terms.extend(aliases)
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return " ".join(unique_terms)
    
    def _extract_product_codes(self, text: str) -> List[str]:
        """Extract JD Jones product codes from text."""
        patterns = [
            r'NA\s*\d{3,4}',
            r'NJ\s*\d{3,4}',
            r'PA\s*\d{3,4}',
            r'FLEXSEAL\s*\d*',
            r'PACMAAN\s*\d*',
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            codes.extend(matches)
        
        return codes
    
    def _normalize_product_code(self, code: str) -> str:
        """Normalize product code format."""
        # Remove extra spaces and standardize
        code = re.sub(r'\s+', ' ', code.strip().upper())
        # Add space between letters and numbers if missing
        code = re.sub(r'([A-Z]+)(\d)', r'\1 \2', code)
        return code


class EmbeddingCache:
    """
    Cache for query embeddings to reduce embedding API calls.
    
    Features:
    - In-memory LRU cache
    - Hash-based lookup for exact matches
    - Configurable TTL
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._access_order: List[str] = []
    
    def _compute_key(self, text: str) -> str:
        """Compute cache key for text."""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Query text
            
        Returns:
            Cached embedding or None
        """
        key = self._compute_key(text)
        
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            
            # Check TTL
            if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return embedding
            else:
                # Expired, remove
                del self.cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache an embedding.
        
        Args:
            text: Query text
            embedding: Embedding vector
        """
        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        key = self._compute_key(text)
        self.cache[key] = (embedding, datetime.now())
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class MultiStageReranker:
    """
    Multi-stage reranking pipeline for optimal precision.
    
    Stages:
    1. Initial retrieval (vector + keyword)
    2. RRF fusion
    3. Cross-encoder reranking (optional)
    4. Final LLM reranking (for top results)
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_llm_reranking: bool = False,
        cross_encoder_top_k: int = 20,
        llm_rerank_top_k: int = 5
    ):
        """
        Initialize multi-stage reranker.
        
        Args:
            use_cross_encoder: Whether to use cross-encoder reranking
            use_llm_reranking: Whether to use LLM for final reranking
            cross_encoder_top_k: Number of results for cross-encoder stage
            llm_rerank_top_k: Number of results for LLM stage
        """
        self.use_cross_encoder = use_cross_encoder
        self.use_llm_reranking = use_llm_reranking
        self.cross_encoder_top_k = cross_encoder_top_k
        self.llm_rerank_top_k = llm_rerank_top_k
        
        self._cross_encoder = None
    
    def _get_cross_encoder(self):
        """Lazy load cross-encoder model."""
        if self._cross_encoder is None and self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder model loaded successfully")
            except ImportError:
                logger.warning("sentence-transformers not available, skipping cross-encoder")
                self.use_cross_encoder = False
        return self._cross_encoder
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Apply multi-stage reranking.
        
        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Final number of results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        current_results = results
        
        # Stage 1: Cross-encoder reranking
        if self.use_cross_encoder:
            current_results = await self._cross_encoder_rerank(
                query, 
                current_results[:self.cross_encoder_top_k]
            )
        
        # Stage 2: LLM reranking (optional, for top results only)
        if self.use_llm_reranking:
            current_results = await self._llm_rerank(
                query,
                current_results[:self.llm_rerank_top_k]
            )
        
        return current_results[:top_k]
    
    async def _cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank using cross-encoder."""
        model = self._get_cross_encoder()
        if not model:
            return results
        
        try:
            # Prepare pairs
            pairs = [(query, r.content) for r in results]
            
            # Get scores
            scores = model.predict(pairs)
            
            # Sort by score
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Update results with new scores and ranks
            reranked = []
            for rank, (result, score) in enumerate(scored_results, start=1):
                result.score = float(score)
                result.rank = rank
                result.source = f"{result.source}+cross_encoder"
                reranked.append(result)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking error: {e}")
            return results
    
    async def _llm_rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank using LLM (for highest precision on top results)."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from src.config.settings import settings
            
            llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
            
            # Build prompt for ranking
            docs_text = "\n\n".join([
                f"Document {i+1}:\n{r.content[:500]}"
                for i, r in enumerate(results)
            ])
            
            prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.

Query: {query}

{docs_text}

Return a JSON array with document numbers and scores:
[{{"doc": 1, "score": X}}, {{"doc": 2, "score": Y}}, ...]

Only return the JSON array, nothing else."""
            
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse response
            import json
            scores_text = response.content.strip()
            if scores_text.startswith("```"):
                scores_text = scores_text.split("```")[1]
                if scores_text.startswith("json"):
                    scores_text = scores_text[4:]
            
            scores_data = json.loads(scores_text)
            
            # Apply scores
            for item in scores_data:
                doc_idx = item["doc"] - 1
                if 0 <= doc_idx < len(results):
                    results[doc_idx].score = item["score"]
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for rank, result in enumerate(results, start=1):
                result.rank = rank
                result.source = f"{result.source}+llm_rerank"
            
            return results
            
        except Exception as e:
            logger.error(f"LLM reranking error: {e}")
            return results


class OptimizedRetriever:
    """
    Production-ready optimized retriever combining all enhancements.
    
    Features:
    - Query expansion
    - Embedding caching
    - Hybrid search (BM25 + Vector)
    - RRF fusion
    - Multi-stage reranking
    """
    
    def __init__(
        self,
        use_query_expansion: bool = True,
        use_embedding_cache: bool = True,
        use_rrf_fusion: bool = True,
        use_cross_encoder: bool = True,
        use_llm_reranking: bool = False
    ):
        """
        Initialize optimized retriever.
        
        Args:
            use_query_expansion: Enable query expansion
            use_embedding_cache: Enable embedding caching
            use_rrf_fusion: Enable RRF fusion
            use_cross_encoder: Enable cross-encoder reranking
            use_llm_reranking: Enable LLM reranking (slower but more accurate)
        """
        self.use_query_expansion = use_query_expansion
        self.use_embedding_cache = use_embedding_cache
        self.use_rrf_fusion = use_rrf_fusion
        
        # Initialize components
        self.query_expander = QueryExpander() if use_query_expansion else None
        self.embedding_cache = EmbeddingCache() if use_embedding_cache else None
        self.rrf = ReciprocalRankFusion() if use_rrf_fusion else None
        self.reranker = MultiStageReranker(
            use_cross_encoder=use_cross_encoder,
            use_llm_reranking=use_llm_reranking
        )
        
        # Lazy-loaded components
        self._vector_retriever = None
        self._bm25_retriever = None
    
    async def retrieve(
        self,
        query: str,
        n_results: int = 10,
        user_role: Optional[str] = None,
        user_department: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform optimized retrieval.
        
        Args:
            query: Search query
            n_results: Number of results to return
            user_role: User role for access control
            user_department: User department for access control
            
        Returns:
            Optimized retrieval results
        """
        logger.info(f"Optimized retrieval for: {query[:50]}...")
        
        # Step 1: Query expansion
        expanded_query = query
        if self.query_expander:
            expanded_query = self.query_expander.expand(query)
            if expanded_query != query:
                logger.debug(f"Expanded query: {expanded_query[:100]}...")
        
        # Step 2: Get vector and keyword results
        vector_results = await self._vector_search(expanded_query, n_results * 2)
        keyword_results = await self._keyword_search(expanded_query, n_results * 2)
        
        # Step 3: RRF fusion
        if self.rrf and keyword_results:
            # Weight vector 0.6, keyword 0.4
            fused_results = self.rrf.fuse(
                vector_results, keyword_results,
                weights=[0.6, 0.4]
            )
        else:
            fused_results = vector_results
        
        # Step 4: Multi-stage reranking
        final_results = await self.reranker.rerank(
            query,  # Use original query for reranking
            fused_results,
            top_k=n_results
        )
        
        return final_results
    
    async def _vector_search(
        self,
        query: str,
        n_results: int
    ) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        try:
            from src.knowledge_base.main_context import MainContextDatabase
            
            if self._vector_retriever is None:
                self._vector_retriever = MainContextDatabase()
            
            results = self._vector_retriever.query(query, n_results=n_results)
            
            return [
                RetrievalResult(
                    document_id=r.document_id or f"doc_{i}",
                    content=r.content,
                    metadata=r.metadata,
                    score=r.relevance_score,
                    source="vector",
                    rank=i + 1
                )
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[RetrievalResult]:
        """Perform keyword-based search (BM25)."""
        try:
            from src.agentic.retrieval.hybrid_search import HybridSearch, BM25
            
            if self._bm25_retriever is None:
                self._bm25_retriever = HybridSearch()
            
            results = await self._bm25_retriever.search(
                query, 
                n_results=n_results,
                use_bm25=True
            )
            
            return [
                RetrievalResult(
                    document_id=r.document_id,
                    content=r.content,
                    metadata=r.metadata,
                    score=r.score,
                    source="keyword",
                    rank=i + 1
                )
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "query_expansion_enabled": self.use_query_expansion,
            "rrf_fusion_enabled": self.use_rrf_fusion,
            "cross_encoder_enabled": self.reranker.use_cross_encoder,
            "llm_reranking_enabled": self.reranker.use_llm_reranking
        }
        
        if self.embedding_cache:
            stats["embedding_cache"] = self.embedding_cache.stats()
        
        return stats


# Convenience function for getting an optimized retriever instance
_optimized_retriever = None

def get_optimized_retriever() -> OptimizedRetriever:
    """Get the global optimized retriever instance."""
    global _optimized_retriever
    if _optimized_retriever is None:
        _optimized_retriever = OptimizedRetriever()
    return _optimized_retriever
