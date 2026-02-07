"""
Reranker
Re-orders search results for improved precision.
Uses cross-encoder, Cohere, or LLM-based reranking.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

from src.config.settings import settings
from src.agentic.retrieval.hybrid_search import SearchResult


@dataclass
class RankedResult:
    """A reranked search result."""
    original_result: SearchResult
    rerank_score: float
    relevance_explanation: str
    final_rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.original_result.document_id,
            "content": self.original_result.content,
            "original_score": self.original_result.score,
            "rerank_score": self.rerank_score,
            "final_rank": self.final_rank,
            "relevance_explanation": self.relevance_explanation,
            "source": self.original_result.source,
            "metadata": self.original_result.metadata
        }


class CohereReranker:
    """
    Cohere Reranker for high-precision reranking.
    
    Uses Cohere's specialized reranking model which is:
    - Faster than LLM-based reranking
    - More accurate than rule-based
    - Specifically trained for relevance scoring
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Reranking model to use
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError("cohere package not installed. Run: pip install cohere")
        return self._client
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank results using Cohere.
        
        Args:
            query: Search query
            results: Results to rerank
            top_k: Number of top results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        if not self.api_key:
            # Fallback to original order if no API key
            return [
                RankedResult(
                    original_result=r,
                    rerank_score=r.score,
                    relevance_explanation="Cohere API key not configured",
                    final_rank=i + 1
                )
                for i, r in enumerate(results[:top_k])
            ]
        
        try:
            client = self._get_client()
            
            # Prepare documents for Cohere
            documents = [r.content for r in results]
            
            # Call Cohere rerank API
            response = client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k
            )
            
            # Build reranked results
            ranked = []
            for i, rerank_result in enumerate(response.results):
                original_idx = rerank_result.index
                ranked.append(RankedResult(
                    original_result=results[original_idx],
                    rerank_score=rerank_result.relevance_score,
                    relevance_explanation=f"Cohere relevance: {rerank_result.relevance_score:.4f}",
                    final_rank=i + 1
                ))
            
            return ranked
            
        except Exception as e:
            logger.error(f"Cohere rerank error: {e}")
            # Fallback to original order
            return [
                RankedResult(
                    original_result=r,
                    rerank_score=r.score,
                    relevance_explanation=f"Cohere error: {str(e)}",
                    final_rank=i + 1
                )
                for i, r in enumerate(results[:top_k])
            ]


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    
    Provides high-quality reranking using models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-large
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Lazy initialization of cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        return self._model
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank using cross-encoder.
        
        Args:
            query: Search query
            results: Results to rerank
            top_k: Number of results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        try:
            model = self._get_model()
            
            # Prepare pairs for cross-encoder
            pairs = [(query, r.content) for r in results]
            
            # Get scores
            scores = model.predict(pairs)
            
            # Create ranked results
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            ranked = []
            for i, (result, score) in enumerate(scored_results[:top_k]):
                ranked.append(RankedResult(
                    original_result=result,
                    rerank_score=float(score),
                    relevance_explanation=f"Cross-encoder score: {score:.4f}",
                    final_rank=i + 1
                ))
            
            return ranked
            
        except Exception as e:
            logger.error(f"Cross-encoder error: {e}")
            return [
                RankedResult(
                    original_result=r,
                    rerank_score=r.score,
                    relevance_explanation=f"Cross-encoder error: {str(e)}",
                    final_rank=i + 1
                )
                for i, r in enumerate(results[:top_k])
            ]


class Reranker:
    """
    Reranks search results for improved precision.
    
    Methods:
    1. cohere: Uses Cohere's specialized reranking model (recommended)
    2. cross_encoder: Uses sentence-transformers cross-encoder
    3. llm: Uses GPT-4 to score relevance
    4. rule_based: Uses domain-specific rules
    5. hybrid: Combines multiple methods
    """
    
    RERANK_PROMPT = """Rate the relevance of this document to the query on a scale of 0-10.

Query: {query}

Document:
{document}

Consider:
1. Does it directly answer the query?
2. Is the information current and accurate?
3. Does it contain specific details (specs, numbers)?
4. Would a user find this helpful?

Respond with ONLY a JSON object:
{{"score": 0-10, "explanation": "brief reason"}}
"""

    def __init__(
        self,
        method: str = "llm",  # llm, rule_based, or hybrid
        model: str = None
    ):
        """
        Initialize reranker.
        
        Args:
            method: Reranking method
            model: LLM model to use
        """
        self.method = method
        
        if method in ["llm", "hybrid"]:
            self.llm = ChatOpenAI(
                model=model or "gpt-4-turbo-preview",
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank search results.
        
        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            List of RankedResult objects
        """
        if self.method == "llm":
            return await self._rerank_llm(query, results, top_k)
        elif self.method == "rule_based":
            return self._rerank_rules(query, results, top_k)
        else:  # hybrid
            llm_results = await self._rerank_llm(query, results, top_k * 2)
            rule_results = self._rerank_rules(query, results, top_k * 2)
            return self._combine_reranks(llm_results, rule_results, top_k)
    
    async def _rerank_llm(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[RankedResult]:
        """Rerank using LLM."""
        import json
        
        ranked = []
        
        # Process in batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            
            tasks = []
            for result in batch:
                prompt = self.RERANK_PROMPT.format(
                    query=query,
                    document=result.content[:1500]  # Truncate long docs
                )
                tasks.append(self._score_document(prompt))
            
            scores = await asyncio.gather(*tasks)
            
            for result, (score, explanation) in zip(batch, scores):
                ranked.append(RankedResult(
                    original_result=result,
                    rerank_score=score,
                    relevance_explanation=explanation,
                    final_rank=0  # Will be set after sorting
                ))
        
        # Sort by rerank score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Set final ranks
        for i, r in enumerate(ranked[:top_k]):
            r.final_rank = i + 1
        
        return ranked[:top_k]
    
    async def _score_document(self, prompt: str) -> tuple:
        """Score a single document."""
        import json
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            
            content = response.content
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                result = json.loads(content[start:end])
                return result.get("score", 5), result.get("explanation", "")
        except Exception as e:
            logger.error(f"Rerank error: {e}")
        
        return 5.0, "Unable to evaluate"
    
    def _rerank_rules(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[RankedResult]:
        """Rerank using domain-specific rules."""
        query_lower = query.lower()
        
        ranked = []
        for result in results:
            score = result.score * 10  # Start with original score
            explanation_parts = []
            
            content_lower = result.content.lower()
            
            # Rule 1: Exact query term matches
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            if term_matches > 0:
                score += term_matches * 0.5
                explanation_parts.append(f"{term_matches} query terms found")
            
            # Rule 2: Product codes mentioned
            product_codes = ["pacmaan", "flexseal", "expansoflex"]
            for code in product_codes:
                if code in query_lower and code in content_lower:
                    score += 2
                    explanation_parts.append(f"Product {code} mentioned")
            
            # Rule 3: Standards mentioned
            standards = ["api 622", "api 624", "shell spe", "fda", "asme"]
            for std in standards:
                if std in query_lower and std in content_lower:
                    score += 1.5
                    explanation_parts.append(f"Standard {std} referenced")
            
            # Rule 4: Technical specifications present
            spec_patterns = ["°c", "bar", "psi", "temperature", "pressure", "rating"]
            if any(p in content_lower for p in spec_patterns):
                score += 1
                explanation_parts.append("Contains specifications")
            
            # Rule 5: Recency boost for datasheets
            if result.metadata.get("document_type") == "datasheet":
                score += 0.5
                explanation_parts.append("Datasheet document")
            
            # Rule 6: Penalty for very short content
            if len(result.content) < 100:
                score -= 1
                explanation_parts.append("Short content")
            
            ranked.append(RankedResult(
                original_result=result,
                rerank_score=min(score, 10),  # Cap at 10
                relevance_explanation="; ".join(explanation_parts) if explanation_parts else "Rule-based scoring",
                final_rank=0
            ))
        
        # Sort by score
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Set ranks
        for i, r in enumerate(ranked[:top_k]):
            r.final_rank = i + 1
        
        return ranked[:top_k]
    
    def _combine_reranks(
        self,
        llm_results: List[RankedResult],
        rule_results: List[RankedResult],
        top_k: int
    ) -> List[RankedResult]:
        """Combine LLM and rule-based reranking."""
        # Create score maps
        llm_scores = {r.original_result.document_id: r.rerank_score for r in llm_results}
        rule_scores = {r.original_result.document_id: r.rerank_score for r in rule_results}
        
        # Combine with weights
        llm_weight = 0.7
        rule_weight = 0.3
        
        combined = []
        seen = set()
        
        for r in llm_results + rule_results:
            doc_id = r.original_result.document_id
            if doc_id in seen:
                continue
            seen.add(doc_id)
            
            llm_score = llm_scores.get(doc_id, 5)
            rule_score = rule_scores.get(doc_id, 5)
            
            combined_score = llm_weight * llm_score + rule_weight * rule_score
            
            combined.append(RankedResult(
                original_result=r.original_result,
                rerank_score=combined_score,
                relevance_explanation=f"Combined: LLM={llm_score:.1f}, Rules={rule_score:.1f}",
                final_rank=0
            ))
        
        combined.sort(key=lambda x: x.rerank_score, reverse=True)
        
        for i, r in enumerate(combined[:top_k]):
            r.final_rank = i + 1
        
        return combined[:top_k]
