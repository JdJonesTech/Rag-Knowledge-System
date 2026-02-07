"""
Optimized BM25 and Inverted Index
Pre-computed term frequencies for O(k) search instead of O(n*m).
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class IndexedDocument:
    """Document with pre-computed term data."""
    doc_id: str
    content: str
    tokens: List[str] = field(default_factory=list)
    tf: Dict[str, int] = field(default_factory=dict)
    length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvertedIndex:
    """
    Inverted index for efficient keyword search.
    
    Provides O(k) search where k = avg postings per query term,
    instead of O(n*m) for brute force.
    """
    
    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(doc_idx, tf)]
        self.documents: List[IndexedDocument] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.avg_doc_length: float = 0
        self.stopwords = stopwords or self._default_stopwords()
    
    def _default_stopwords(self) -> Set[str]:
        return {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'and', 'or', 'but', 'if',
                'then', 'else', 'when', 'at', 'by', 'for', 'with', 'about', 'against',
                'between', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                'under', 'again', 'further', 'once', 'of', 'this', 'that', 'these', 'those'}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with optional stopword removal."""
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a document to the index."""
        tokens = self.tokenize(content)
        tf = Counter(tokens)
        
        doc = IndexedDocument(
            doc_id=doc_id, content=content, tokens=tokens,
            tf=dict(tf), length=len(tokens), metadata=metadata or {}
        )
        doc_idx = len(self.documents)
        self.documents.append(doc)
        
        for term, freq in tf.items():
            self.index[term].append((doc_idx, freq))
            self.doc_freqs[term] += 1
        
        total_length = sum(d.length for d in self.documents)
        self.avg_doc_length = total_length / len(self.documents)
    
    def build_from_documents(self, documents: List[Dict[str, Any]]):
        """Build index from list of documents."""
        for doc in documents:
            self.add_document(
                doc_id=doc.get('document_id', str(len(self.documents))),
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {})
            )
        logger.info(f"Built inverted index with {len(self.documents)} docs, {len(self.index)} terms")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using inverted index. Returns (doc_idx, score) tuples."""
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        scores: Dict[int, float] = defaultdict(float)
        for term in query_tokens:
            if term not in self.index:
                continue
            for doc_idx, _ in self.index[term]:
                scores[doc_idx] += 1  # Simple TF scoring
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class OptimizedBM25:
    """
    Optimized BM25 with pre-computed term frequencies.
    
    Key optimizations:
    1. Pre-compute TF for each document during indexing
    2. Use inverted index for query terms
    3. Cache IDF values
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = InvertedIndex()
        self.idf: Dict[str, float] = {}
        self.indexed = False
    
    def index(self, documents: List[Dict[str, Any]]):
        """Build index from documents."""
        self.inverted_index.build_from_documents(documents)
        n_docs = len(self.inverted_index.documents)
        
        # Pre-compute IDF for all terms
        for term, df in self.inverted_index.doc_freqs.items():
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        self.indexed = True
        logger.info(f"OptimizedBM25 indexed {n_docs} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search with BM25 scoring."""
        if not self.indexed:
            return []
        
        query_tokens = self.inverted_index.tokenize(query)
        if not query_tokens:
            return []
        
        scores: Dict[int, float] = defaultdict(float)
        avg_dl = self.inverted_index.avg_doc_length
        
        for term in query_tokens:
            if term not in self.inverted_index.index:
                continue
            
            idf = self.idf.get(term, 0)
            for doc_idx, tf in self.inverted_index.index[term]:
                doc = self.inverted_index.documents[doc_idx]
                doc_length = doc.length
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_dl)
                scores[doc_idx] += idf * (numerator / denominator)
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_document(self, doc_idx: int) -> Optional[IndexedDocument]:
        """Get document by index."""
        if 0 <= doc_idx < len(self.inverted_index.documents):
            return self.inverted_index.documents[doc_idx]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': len(self.inverted_index.documents),
            'num_terms': len(self.inverted_index.index),
            'avg_doc_length': self.inverted_index.avg_doc_length,
            'total_postings': sum(len(v) for v in self.inverted_index.index.values())
        }
