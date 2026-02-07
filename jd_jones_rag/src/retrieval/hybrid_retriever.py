"""
Hybrid Retriever Module for RAG-based product knowledge retrieval.

This module provides a hybrid search implementation for quotation processing,
combining vector search with keyword matching for optimal results.
"""

from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever that combines vector and keyword search."""
    
    def __init__(self):
        self._search = None
    
    def _get_search(self):
        """Lazy load the hybrid search to avoid import issues at startup."""
        if self._search is None:
            try:
                from src.agentic.retrieval.hybrid_search import HybridSearch
                self._search = HybridSearch()
            except ImportError:
                logger.warning("HybridSearch not available, using fallback")
                self._search = FallbackSearch()
        return self._search
    
    async def aretrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Async retrieve documents matching the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of document results
        """
        search = self._get_search()
        
        try:
            import asyncio
            import inspect
            
            # Try async methods first
            if hasattr(search, 'asearch'):
                result = search.asearch(query, top_k=top_k)
                if inspect.iscoroutine(result):
                    return await result
                return result
            elif hasattr(search, 'search'):
                result = search.search(query, top_k=top_k)
                # Check if result is a coroutine and await it
                if inspect.iscoroutine(result):
                    return await result
                return result if result else []
            elif hasattr(search, 'aretrieve'):
                result = search.aretrieve(query, top_k=top_k)
                if inspect.iscoroutine(result):
                    return await result
                return result
            else:
                logger.warning("No search method found on HybridSearch")
                return []
        except Exception as e:
            logger.warning(f"Hybrid retrieval failed: {e}")
            return []
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Sync retrieve documents matching the query."""
        search = self._get_search()
        
        try:
            if hasattr(search, 'search'):
                return search.search(query, top_k=top_k)
            elif hasattr(search, 'retrieve'):
                return search.retrieve(query, top_k=top_k)
            else:
                logger.warning("No search method found on HybridSearch")
                return []
        except Exception as e:
            logger.warning(f"Hybrid retrieval failed: {e}")
            return []


class FallbackSearch:
    """Fallback search when hybrid search is not available."""
    
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """No-op fallback search."""
        logger.info(f"Fallback search for: {query[:50]}...")
        return []
    
    async def aretrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """No-op fallback async search."""
        return self.search(query, top_k)


# Singleton instance
_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get the singleton hybrid retriever instance.
    
    Returns:
        HybridRetriever instance for product knowledge retrieval.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
