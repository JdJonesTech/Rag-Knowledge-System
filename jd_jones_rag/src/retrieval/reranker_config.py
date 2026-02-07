"""
Production Reranking Configuration
Configures cross-encoder and multi-stage reranking for production use.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for production reranking."""
    
    # Cross-encoder settings
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: int = 20  # Rerank top 20 results
    cross_encoder_batch_size: int = 8
    
    # LLM reranking settings (optional, slower but more accurate)
    use_llm_reranking: bool = False
    llm_rerank_top_k: int = 5  # Only LLM-rerank top 5
    llm_model: str = "gpt-4-turbo-preview"
    
    # RRF settings
    use_rrf: bool = True
    rrf_k: int = 60  # Standard RRF constant
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # Query expansion
    use_query_expansion: bool = True
    
    # Caching
    use_embedding_cache: bool = True
    use_faq_cache: bool = True
    embedding_cache_ttl_hours: int = 168  # 1 week
    faq_cache_ttl_hours: int = 168  # 1 week
    
    # Performance thresholds
    min_relevance_score: float = 0.3
    max_results: int = 10
    
    # Hardware optimization
    device: str = "cuda"  # "cuda", "cpu", "mps"
    half_precision: bool = True  # Use FP16 on GPU


# Production configuration presets
PRODUCTION_CONFIG = RerankerConfig(
    use_cross_encoder=True,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder_top_k=20,
    use_llm_reranking=False,  # Disabled by default for latency
    use_rrf=True,
    vector_weight=0.6,
    keyword_weight=0.4,
    use_query_expansion=True,
    use_embedding_cache=True,
    use_faq_cache=True,
    device="cuda",
    half_precision=True
)

DEVELOPMENT_CONFIG = RerankerConfig(
    use_cross_encoder=True,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder_top_k=10,
    use_llm_reranking=False,
    use_rrf=True,
    vector_weight=0.5,
    keyword_weight=0.5,
    use_query_expansion=True,
    use_embedding_cache=True,
    use_faq_cache=True,
    device="cpu",  # CPU for development
    half_precision=False
)

HIGH_ACCURACY_CONFIG = RerankerConfig(
    use_cross_encoder=True,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger model
    cross_encoder_top_k=30,
    use_llm_reranking=True,  # Enable LLM reranking for max accuracy
    llm_rerank_top_k=5,
    llm_model="gpt-4-turbo-preview",
    use_rrf=True,
    vector_weight=0.6,
    keyword_weight=0.4,
    use_query_expansion=True,
    use_embedding_cache=True,
    use_faq_cache=True,
    device="cuda",
    half_precision=True
)

LOW_LATENCY_CONFIG = RerankerConfig(
    use_cross_encoder=False,  # Skip for latency
    use_llm_reranking=False,
    use_rrf=True,
    vector_weight=0.7,
    keyword_weight=0.3,
    use_query_expansion=False,  # Skip for latency
    use_embedding_cache=True,  # Important for latency
    use_faq_cache=True,  # Important for latency
    device="cpu",
    half_precision=False
)


def get_config(preset: str = "production") -> RerankerConfig:
    """
    Get reranking configuration by preset name.
    
    Args:
        preset: Configuration preset name
        
    Returns:
        RerankerConfig instance
    """
    configs = {
        "production": PRODUCTION_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "high_accuracy": HIGH_ACCURACY_CONFIG,
        "low_latency": LOW_LATENCY_CONFIG
    }
    
    if preset not in configs:
        logger.warning(f"Unknown config preset '{preset}', using production")
        return PRODUCTION_CONFIG
    
    return configs[preset]


class ProductionReranker:
    """
    Production-ready reranker with configurable cross-encoder.
    
    Usage:
        from src.retrieval.reranker_config import ProductionReranker
        
        reranker = ProductionReranker("production")
        results = await reranker.rerank(query, documents)
    """
    
    def __init__(self, config_preset: str = "production"):
        """
        Initialize production reranker.
        
        Args:
            config_preset: Configuration preset name
        """
        self.config = get_config(config_preset)
        self._cross_encoder = None
        self._llm = None
        
        logger.info(f"ProductionReranker initialized with '{config_preset}' config")
        logger.info(f"  - Cross-encoder: {self.config.use_cross_encoder}")
        logger.info(f"  - LLM reranking: {self.config.use_llm_reranking}")
        logger.info(f"  - RRF enabled: {self.config.use_rrf}")
    
    def _get_cross_encoder(self):
        """Lazy load cross-encoder model."""
        if self._cross_encoder is None and self.config.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                import torch
                
                device = self.config.device
                if device == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = "cpu"
                
                self._cross_encoder = CrossEncoder(
                    self.config.cross_encoder_model,
                    device=device
                )
                
                if self.config.half_precision and device == "cuda":
                    self._cross_encoder.model.half()
                
                logger.info(f"Cross-encoder loaded: {self.config.cross_encoder_model} on {device}")
                
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                self.config.use_cross_encoder = False
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self.config.use_cross_encoder = False
        
        return self._cross_encoder
    
    async def rerank(
        self,
        query: str,
        documents: list,
        top_k: Optional[int] = None
    ) -> list:
        """
        Rerank documents using configured strategy.
        
        Args:
            query: Search query
            documents: List of documents (dicts with 'content' key)
            top_k: Number of results to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        top_k = top_k or self.config.max_results
        current_docs = documents[:self.config.cross_encoder_top_k]
        
        # Stage 1: Cross-encoder reranking
        if self.config.use_cross_encoder:
            current_docs = self._cross_encoder_rerank(query, current_docs)
        
        # Stage 2: LLM reranking (optional)
        if self.config.use_llm_reranking:
            current_docs = await self._llm_rerank(
                query, 
                current_docs[:self.config.llm_rerank_top_k]
            )
        
        # Filter by minimum score
        current_docs = [
            d for d in current_docs 
            if d.get("rerank_score", 1.0) >= self.config.min_relevance_score
        ]
        
        return current_docs[:top_k]
    
    def _cross_encoder_rerank(self, query: str, documents: list) -> list:
        """Apply cross-encoder reranking."""
        model = self._get_cross_encoder()
        if not model:
            return documents
        
        try:
            # Prepare query-document pairs
            pairs = [
                (query, doc.get("content", "")[:512])  # Limit content length
                for doc in documents
            ]
            
            # Get scores
            scores = model.predict(
                pairs, 
                batch_size=self.config.cross_encoder_batch_size,
                show_progress_bar=False
            )
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by score
            documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            logger.debug(f"Cross-encoder reranked {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents
    
    async def _llm_rerank(self, query: str, documents: list) -> list:
        """Apply LLM-based reranking."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from src.config.settings import settings
            
            if self._llm is None:
                self._llm = ChatOpenAI(
                    model=self.config.llm_model,
                    temperature=0,
                    openai_api_key=settings.openai_api_key
                )
            
            # Build prompt
            docs_text = "\n\n".join([
                f"Document {i+1}:\n{doc.get('content', '')[:400]}"
                for i, doc in enumerate(documents)
            ])
            
            prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.

Query: {query}

{docs_text}

Return a JSON array with document numbers and scores:
[{{"doc": 1, "score": X}}, {{"doc": 2, "score": Y}}, ...]

Only return the JSON array."""
            
            response = await self._llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse response
            import json
            import re
            
            content = response.content.strip()
            # Extract JSON from possible markdown
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                scores_data = json.loads(json_match.group())
                
                for item in scores_data:
                    doc_idx = item["doc"] - 1
                    if 0 <= doc_idx < len(documents):
                        documents[doc_idx]["llm_score"] = item["score"]
                        # Combine with cross-encoder score
                        ce_score = documents[doc_idx].get("rerank_score", 5.0)
                        documents[doc_idx]["rerank_score"] = (ce_score + item["score"]) / 2
                
                documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            logger.debug(f"LLM reranked {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return documents
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration info."""
        return {
            "use_cross_encoder": self.config.use_cross_encoder,
            "cross_encoder_model": self.config.cross_encoder_model,
            "cross_encoder_loaded": self._cross_encoder is not None,
            "use_llm_reranking": self.config.use_llm_reranking,
            "use_rrf": self.config.use_rrf,
            "device": self.config.device,
            "half_precision": self.config.half_precision
        }


# Global instance for singleton access
_production_reranker = None

def get_production_reranker(config_preset: str = "production") -> ProductionReranker:
    """Get global production reranker instance."""
    global _production_reranker
    if _production_reranker is None:
        _production_reranker = ProductionReranker(config_preset)
    return _production_reranker
