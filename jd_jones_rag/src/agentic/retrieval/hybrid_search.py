"""
Hybrid Search
Combines BM25 (keyword-based) search with vector (semantic) search.
Provides higher precision for technical and legal documents.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

from src.config.settings import settings


@dataclass
class SearchResult:
    """A single search result."""
    document_id: str
    content: str
    score: float
    vector_score: float
    bm25_score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "source": self.source,
            "metadata": self.metadata
        }


class BM25:
    """
    BM25 (Best Match 25) ranking algorithm.
    Classic keyword-based search algorithm.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.idf: Dict[str, float] = {}
        self.tokenized_docs: List[List[str]] = []
        # OPTIMIZATION: Pre-computed TF for each document
        self.tf_counters: List[Counter] = []
        # OPTIMIZATION: Inverted index for faster term lookup
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 search.
        
        OPTIMIZATION: Pre-computes term frequencies and builds inverted index
        during indexing for O(k) search instead of O(n*m).
        
        Args:
            documents: List of documents with 'content' field
        """
        self.documents = documents
        self.tokenized_docs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()
        self.tf_counters = []  # Pre-computed TF
        self.inverted_index = {}  # Inverted index
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc.get("content", ""))
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Pre-compute TF for this document
            tf_counter = Counter(tokens)
            self.tf_counters.append(tf_counter)
            
            # Build inverted index
            for token, freq in tf_counter.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_idx, freq))
            
            # Count document frequencies (how many docs contain each term)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF for each term
        n_docs = len(documents)
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents using BM25.
        
        OPTIMIZATION: Uses inverted index and pre-computed TF for O(k) search
        instead of O(n*m) where n=docs and m=query terms.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores: Dict[int, float] = {}
        
        # Use inverted index for efficient search
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self.idf.get(term, 0)
            
            # Iterate only over documents containing this term
            for doc_idx, tf in self.inverted_index[term]:
                doc_length = self.doc_lengths[doc_idx]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                term_score = idf * (numerator / denominator)
                
                if doc_idx in scores:
                    scores[doc_idx] += term_score
                else:
                    scores[doc_idx] = term_score
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


class HybridSearch:
    """
    Hybrid search combining BM25 and vector search.
    
    Benefits:
    - BM25 excels at exact keyword matches (technical terms, product codes)
    - Vector search excels at semantic understanding
    - Combined: Best of both worlds
    """
    
    def __init__(
        self,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        retriever=None
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_weight: Weight for vector search scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            retriever: Vector store retriever
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.retriever = retriever
        self.bm25 = BM25()
        self.documents: List[Dict[str, Any]] = []
        self.indexed = False
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of documents with content and metadata
        """
        self.documents = documents
        self.bm25.index(documents)
        self.indexed = True
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        if not self.indexed:
            raise ValueError("Documents must be indexed first")
        
        # Get BM25 results
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Normalize BM25 scores to 0-1
        max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1
        bm25_scores = {idx: score / max_bm25 if max_bm25 > 0 else 0 for idx, score in bm25_results}
        
        # Get vector search results
        vector_scores = {}
        if self.retriever:
            try:
                from src.knowledge_base.retriever import UserRole
                vector_results = self.retriever.retrieve(
                    query=query,
                    user_role=UserRole.ADMIN,
                    n_results=top_k * 2
                )
                
                # OPTIMIZATION: Build hash maps for O(1) lookup instead of O(n) loop
                content_to_idx = {doc.get("content", "")[:200]: idx for idx, doc in enumerate(self.documents)}
                docid_to_idx = {doc.get("document_id", ""): idx for idx, doc in enumerate(self.documents) if doc.get("document_id")}
                
                # Single-pass mapping with O(1) lookups
                for result in vector_results.all_results:
                    idx = None
                    # Try document_id first (faster)
                    doc_id = result.metadata.get("document_id")
                    if doc_id and doc_id in docid_to_idx:
                        idx = docid_to_idx[doc_id]
                    # Fallback to content hash
                    elif result.content[:200] in content_to_idx:
                        idx = content_to_idx[result.content[:200]]
                    
                    if idx is not None:
                        vector_scores[idx] = result.relevance_score
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        
        # Combine scores
        combined_results = []
        seen_indices = set()
        
        # Combine indices from both searches
        all_indices = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        for idx in all_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            bm25_score = bm25_scores.get(idx, 0)
            vector_score = vector_scores.get(idx, 0)
            
            # Weighted combination
            combined_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score
            )
            
            doc = self.documents[idx]
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if doc.get("metadata", {}).get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            combined_results.append(SearchResult(
                document_id=doc.get("document_id", str(idx)),
                content=doc.get("content", ""),
                score=combined_score,
                vector_score=vector_score,
                bm25_score=bm25_score,
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {})
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:top_k]
    
    def search_sync(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Synchronous search (BM25 only, no vector).
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of SearchResult objects
        """
        if not self.indexed:
            raise ValueError("Documents must be indexed first")
        
        bm25_results = self.bm25.search(query, top_k=top_k)
        max_score = max(score for _, score in bm25_results) if bm25_results else 1
        
        results = []
        for idx, score in bm25_results:
            doc = self.documents[idx]
            
            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if doc.get("metadata", {}).get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            results.append(SearchResult(
                document_id=doc.get("document_id", str(idx)),
                content=doc.get("content", ""),
                score=score / max_score if max_score > 0 else 0,
                vector_score=0,
                bm25_score=score / max_score if max_score > 0 else 0,
                source=doc.get("source", "unknown"),
                metadata=doc.get("metadata", {})
            ))
        
        return results[:top_k]
    
    def adjust_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """
        Adjust the weights for vector and BM25 scores.
        
        Args:
            vector_weight: New vector weight
            bm25_weight: New BM25 weight
        """
        total = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total
        self.bm25_weight = bm25_weight / total
