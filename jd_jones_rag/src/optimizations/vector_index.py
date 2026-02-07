"""
Optimized Vector Index
Provides fast approximate nearest neighbor search using numpy vectorized operations
with optional FAISS backend for production scale.

OPTIMIZATIONS:
1. Numpy vectorized similarity computation (batch matrix operations)
2. Optional FAISS integration for ANN search
3. Annoy fallback for memory-constrained environments
4. Pre-built index for O(log n) search instead of O(n)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class OptimizedVectorIndex:
    """
    Optimized vector index with multiple backend options.
    
    Backends (in order of preference):
    1. FAISS (if available) - Best performance for large datasets
    2. Annoy (if available) - Good performance, lower memory
    3. Numpy (fallback) - Vectorized operations, no dependencies
    
    Features:
    - Batch vectorized similarity computation
    - Configurable index building
    - Thread-safe operations
    - Incremental updates
    """
    
    def __init__(
        self,
        dimension: int = 384,
        backend: str = "auto",  # auto, faiss, annoy, numpy
        n_trees: int = 10,  # For Annoy
        nlist: int = 100,   # For FAISS IVF
    ):
        self.dimension = dimension
        self.backend = backend
        self.n_trees = n_trees
        self.nlist = nlist
        
        # Storage
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # Pre-computed matrix
        self._matrix_dirty = True  # Flag to rebuild matrix
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Backend-specific index
        self._faiss_index = None
        self._annoy_index = None
        
        # Detect available backends
        self._backend_type = self._detect_backend()
        logger.info(f"OptimizedVectorIndex using backend: {self._backend_type}")
    
    def _detect_backend(self) -> str:
        """Detect best available backend."""
        if self.backend != "auto":
            return self.backend
        
        # Try FAISS first
        try:
            import faiss
            return "faiss"
        except ImportError:
            pass
        
        # Try Annoy
        try:
            from annoy import AnnoyIndex
            return "annoy"
        except ImportError:
            pass
        
        # Fallback to numpy
        return "numpy"
    
    def add(self, id: str, vector: List[float]) -> None:
        """Add a vector to the index."""
        with self._lock:
            arr = np.array(vector, dtype=np.float32)
            
            if id in self._vectors:
                # Update existing
                self._vectors[id] = arr
            else:
                # Add new
                self._vectors[id] = arr
                self._id_to_idx[id] = len(self._idx_to_id)
                self._idx_to_id.append(id)
            
            self._matrix_dirty = True
    
    def add_batch(self, items: Dict[str, List[float]]) -> None:
        """Add multiple vectors at once."""
        with self._lock:
            for id, vector in items.items():
                arr = np.array(vector, dtype=np.float32)
                
                if id not in self._vectors:
                    self._id_to_idx[id] = len(self._idx_to_id)
                    self._idx_to_id.append(id)
                
                self._vectors[id] = arr
            
            self._matrix_dirty = True
    
    def remove(self, id: str) -> bool:
        """Remove a vector from the index."""
        with self._lock:
            if id in self._vectors:
                del self._vectors[id]
                # Note: We don't remove from idx mappings to avoid O(n) rebuild
                # The search will skip missing vectors
                self._matrix_dirty = True
                return True
            return False
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the search matrix (OPTIMIZATION: vectorized operations)."""
        if not self._matrix_dirty:
            return
        
        with self._lock:
            if not self._vectors:
                self._matrix = None
                return
            
            # Build matrix in order of idx_to_id
            valid_vectors = []
            valid_ids = []
            
            for id in self._idx_to_id:
                if id in self._vectors:
                    valid_vectors.append(self._vectors[id])
                    valid_ids.append(id)
            
            if valid_vectors:
                self._matrix = np.vstack(valid_vectors)
                # Normalize for cosine similarity
                norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                self._matrix = self._matrix / norms
                
                # Update mappings to match matrix
                self._idx_to_id = valid_ids
                self._id_to_idx = {id: i for i, id in enumerate(valid_ids)}
            else:
                self._matrix = None
            
            self._matrix_dirty = False
            
            # Rebuild backend index if using FAISS/Annoy
            if self._backend_type == "faiss" and self._matrix is not None:
                self._rebuild_faiss_index()
            elif self._backend_type == "annoy" and self._matrix is not None:
                self._rebuild_annoy_index()
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index."""
        try:
            import faiss
            
            n_vectors = self._matrix.shape[0]
            
            if n_vectors < 100:
                # Use flat index for small datasets
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
            else:
                # Use IVF for larger datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(self.nlist, n_vectors // 10)
                self._faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
                self._faiss_index.train(self._matrix)
            
            self._faiss_index.add(self._matrix)
            logger.debug(f"FAISS index rebuilt with {n_vectors} vectors")
            
        except Exception as e:
            logger.error(f"FAISS index rebuild failed: {e}")
            self._backend_type = "numpy"
    
    def _rebuild_annoy_index(self) -> None:
        """Rebuild Annoy index."""
        try:
            from annoy import AnnoyIndex
            
            self._annoy_index = AnnoyIndex(self.dimension, 'angular')
            
            for i, vector in enumerate(self._matrix):
                self._annoy_index.add_item(i, vector)
            
            self._annoy_index.build(self.n_trees)
            logger.debug(f"Annoy index rebuilt with {len(self._matrix)} vectors")
            
        except Exception as e:
            logger.error(f"Annoy index rebuild failed: {e}")
            self._backend_type = "numpy"
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        OPTIMIZATION: Uses vectorized numpy operations for O(1) per candidate
        instead of O(n) individual comparisons.
        """
        self._rebuild_matrix()
        
        if self._matrix is None or len(self._matrix) == 0:
            return []
        
        query = np.array(query_vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)  # Normalize
        
        if self._backend_type == "faiss":
            return self._search_faiss(query, top_k, min_score)
        elif self._backend_type == "annoy":
            return self._search_annoy(query, top_k, min_score)
        else:
            return self._search_numpy(query, top_k, min_score)
    
    def _search_numpy(
        self,
        query: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """
        OPTIMIZATION: Vectorized numpy search.
        Computes all similarities in a single matrix operation.
        """
        # Batch dot product (all similarities at once)
        similarities = np.dot(self._matrix, query)
        
        # Get top k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append(SearchResult(
                    id=self._idx_to_id[idx],
                    score=score
                ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _search_faiss(
        self,
        query: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """Search using FAISS index."""
        if self._faiss_index is None:
            return self._search_numpy(query, top_k, min_score)
        
        try:
            # FAISS expects 2D array
            query_2d = query.reshape(1, -1)
            scores, indices = self._faiss_index.search(query_2d, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= min_score:
                    results.append(SearchResult(
                        id=self._idx_to_id[idx],
                        score=float(score)
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._search_numpy(query, top_k, min_score)
    
    def _search_annoy(
        self,
        query: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[SearchResult]:
        """Search using Annoy index."""
        if self._annoy_index is None:
            return self._search_numpy(query, top_k, min_score)
        
        try:
            indices, distances = self._annoy_index.get_nns_by_vector(
                query.tolist(), top_k, include_distances=True
            )
            
            results = []
            for idx, dist in zip(indices, distances):
                # Annoy returns angular distance, convert to similarity
                score = 1.0 - (dist / 2.0)  # Approximate conversion
                if score >= min_score:
                    results.append(SearchResult(
                        id=self._idx_to_id[idx],
                        score=score
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Annoy search failed: {e}")
            return self._search_numpy(query, top_k, min_score)
    
    def __len__(self) -> int:
        return len(self._vectors)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "backend": self._backend_type,
            "num_vectors": len(self._vectors),
            "dimension": self.dimension,
            "matrix_dirty": self._matrix_dirty,
            "faiss_available": self._faiss_index is not None,
            "annoy_available": self._annoy_index is not None
        }


# Singleton instance
_vector_index: Optional[OptimizedVectorIndex] = None


def get_vector_index(dimension: int = 384) -> OptimizedVectorIndex:
    """Get singleton vector index instance."""
    global _vector_index
    if _vector_index is None:
        _vector_index = OptimizedVectorIndex(dimension=dimension)
    return _vector_index
