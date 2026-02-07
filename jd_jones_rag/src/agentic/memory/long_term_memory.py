"""
Long-Term Memory
Persistent memory for user preferences, historical data, and learned patterns.
Uses vector-enhanced storage for semantic retrieval.

OPTIMIZATIONS:
- OptimizedVectorIndex for O(log n) ANN search instead of O(n) linear scan
- Numpy vectorized similarity when index not available
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of long-term memories."""
    USER_PREFERENCE = "user_preference"
    INTERACTION_HISTORY = "interaction_history"
    LEARNED_PATTERN = "learned_pattern"
    ENTITY_FACT = "entity_fact"
    SYSTEM_KNOWLEDGE = "system_knowledge"


@dataclass
class Memory:
    """A single long-term memory entry."""
    memory_id: str
    memory_type: MemoryType
    content: str
    user_id: Optional[str]
    embedding: Optional[List[float]] = None
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "user_id": self.user_id,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class UserProfile:
    """User profile built from memories."""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    industries: List[str] = field(default_factory=list)
    products_interested: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "industries": self.industries,
            "products_interested": self.products_interested,
            "interaction_patterns": self.interaction_patterns,
            "last_updated": self.last_updated.isoformat()
        }


class LongTermMemory:
    """
    Manages long-term memory with semantic retrieval.
    
    OPTIMIZATIONS:
    - Uses OptimizedVectorIndex for O(log n) ANN search
    - Falls back to numpy vectorized operations
    
    Features:
    - User preference learning
    - Interaction history tracking
    - Pattern recognition
    - Semantic memory retrieval
    - Memory consolidation
    """
    
    def __init__(
        self,
        embedding_generator=None,
        similarity_threshold: float = 0.8,
        max_memories_per_user: int = 1000
    ):
        """
        Initialize long-term memory.
        
        Args:
            embedding_generator: Embedding generator for semantic search
            similarity_threshold: Threshold for similar memory detection
            max_memories_per_user: Maximum memories per user
        """
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.max_memories_per_user = max_memories_per_user
        
        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.user_memories: Dict[str, List[str]] = {}  # user_id -> memory_ids
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # OPTIMIZATION: Use OptimizedVectorIndex for ANN search
        self._vector_index = None
        try:
            from src.optimizations.vector_index import OptimizedVectorIndex
            self._vector_index = OptimizedVectorIndex(dimension=384)
            logger.info("LongTermMemory using OptimizedVectorIndex for ANN search")
        except ImportError:
            logger.warning("OptimizedVectorIndex not available, using linear search")
        
        # Fallback: Embedding index for linear search
        self.embedding_index: Dict[str, List[float]] = {}
    
    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            user_id: Associated user
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Created Memory object
        """
        # Generate memory ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        memory_id = f"mem_{content_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Check for similar existing memory
        if user_id:
            similar = await self.find_similar(content, user_id, top_k=1)
            if similar and similar[0][1] > self.similarity_threshold:
                # Update existing memory instead
                existing = similar[0][0]
                existing.access_count += 1
                existing.last_accessed = datetime.now()
                existing.importance_score = max(existing.importance_score, importance)
                return existing
        
        # Generate embedding
        embedding = None
        if self.embedding_generator:
            try:
                embedding = await self._get_embedding(content)
            except Exception as e:
                logger.debug(f"Failed to generate embedding for memory: {e}")
        
        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            user_id=user_id,
            embedding=embedding,
            importance_score=importance,
            metadata=metadata or {}
        )
        
        self.memories[memory_id] = memory
        
        if embedding:
            self.embedding_index[memory_id] = embedding
            # OPTIMIZATION: Also add to vector index for ANN search
            if self._vector_index:
                self._vector_index.add(memory_id, embedding)
        
        # Track user memories
        if user_id:
            if user_id not in self.user_memories:
                self.user_memories[user_id] = []
            self.user_memories[user_id].append(memory_id)
            
            # Enforce limit
            if len(self.user_memories[user_id]) > self.max_memories_per_user:
                self._consolidate_user_memories(user_id)
            
            # Update user profile
            self._update_user_profile(user_id, memory)
        
        return memory
    
    async def recall(
        self,
        query: str,
        user_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5
    ) -> List[Memory]:
        """
        Recall relevant memories.
        
        Args:
            query: Query for retrieval
            user_id: Filter by user
            memory_type: Filter by type
            top_k: Number of memories to return
            
        Returns:
            List of relevant memories
        """
        # Get candidate memories
        candidates = list(self.memories.values())
        
        if user_id:
            candidates = [m for m in candidates if m.user_id == user_id or m.user_id is None]
        
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        if not candidates:
            return []
        
        # Semantic search if embedding available
        if self.embedding_generator:
            try:
                query_embedding = await self._get_embedding(query)
                
                # OPTIMIZATION: Use vector index for O(log n) ANN search
                if self._vector_index and len(self._vector_index) > 0:
                    # Filter candidate memory IDs
                    candidate_ids = {m.memory_id for m in candidates}
                    
                    # Search with more results to account for filtering
                    search_results = self._vector_index.search(query_embedding, top_k=top_k * 3)
                    
                    scored = []
                    for result in search_results:
                        if result.id in candidate_ids and result.id in self.memories:
                            memory = self.memories[result.id]
                            scored.append((memory, result.score))
                    
                    # Sort by score (should already be sorted but ensure)
                    scored.sort(key=lambda x: x[1], reverse=True)
                else:
                    # FALLBACK: Vectorized numpy computation (still faster than pure loop)
                    scored = []
                    candidate_ids = [m.memory_id for m in candidates if m.memory_id in self.embedding_index]
                    
                    if candidate_ids:
                        # Build matrix for vectorized computation
                        embeddings = np.array([self.embedding_index[mid] for mid in candidate_ids])
                        query_arr = np.array(query_embedding)
                        
                        # Vectorized cosine similarity
                        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_arr)
                        norms[norms == 0] = 1  # Avoid division by zero
                        similarities = np.dot(embeddings, query_arr) / norms
                        
                        # Get top results
                        for i, mid in enumerate(candidate_ids):
                            scored.append((self.memories[mid], float(similarities[i])))
                        
                        scored.sort(key=lambda x: x[1], reverse=True)
                
                # Update access counts
                results = []
                for memory, _ in scored[:top_k]:
                    memory.access_count += 1
                    memory.last_accessed = datetime.now()
                    results.append(memory)
                
                return results
                
            except Exception as e:
                logger.debug(f"Semantic search failed, falling back to keyword: {e}")
        
        # Fallback: keyword matching
        query_lower = query.lower()
        scored = []
        for memory in candidates:
            content_lower = memory.content.lower()
            # Simple overlap score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((memory, overlap))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]
    
    async def find_similar(
        self,
        content: str,
        user_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[tuple]:
        """Find similar memories."""
        if not self.embedding_generator:
            return []
        
        try:
            content_embedding = await self._get_embedding(content)
            
            candidates = self.memories.values()
            if user_id:
                candidates = [m for m in candidates if m.user_id == user_id]
            
            scored = []
            for memory in candidates:
                if memory.memory_id in self.embedding_index:
                    similarity = self._compute_similarity(
                        content_embedding,
                        self.embedding_index[memory.memory_id]
                    )
                    scored.append((memory, similarity))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
            
        except Exception:
            return []
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.user_profiles[user_id]
    
    def learn_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any
    ):
        """Learn a user preference."""
        profile = self.get_user_profile(user_id)
        profile.preferences[preference_key] = preference_value
        profile.last_updated = datetime.now()
    
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        profile = self.user_profiles.get(user_id)
        return profile.preferences if profile else {}
    
    def _update_user_profile(self, user_id: str, memory: Memory):
        """Update user profile from memory."""
        profile = self.get_user_profile(user_id)
        
        # Extract information based on memory type
        if memory.memory_type == MemoryType.USER_PREFERENCE:
            # Parse preference from content
            content = memory.content.lower()
            if "industry" in content:
                for ind in ["oil", "gas", "chemical", "pharma", "food", "power"]:
                    if ind in content:
                        if ind not in profile.industries:
                            profile.industries.append(ind)
        
        elif memory.memory_type == MemoryType.INTERACTION_HISTORY:
            # Track interaction patterns
            metadata = memory.metadata
            if "product" in metadata:
                if metadata["product"] not in profile.products_interested:
                    profile.products_interested.append(metadata["product"])
        
        profile.last_updated = datetime.now()
    
    def _consolidate_user_memories(self, user_id: str):
        """Consolidate user memories when limit exceeded."""
        if user_id not in self.user_memories:
            return
        
        memory_ids = self.user_memories[user_id]
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        
        # Sort by importance and access
        memories.sort(key=lambda m: (m.importance_score, m.access_count), reverse=True)
        
        # Keep top memories
        keep_count = int(self.max_memories_per_user * 0.8)
        to_keep = set(m.memory_id for m in memories[:keep_count])
        
        # Remove old memories
        for mid in memory_ids:
            if mid not in to_keep:
                if mid in self.memories:
                    del self.memories[mid]
                if mid in self.embedding_index:
                    del self.embedding_index[mid]
        
        self.user_memories[user_id] = [mid for mid in memory_ids if mid in to_keep]
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if hasattr(self.embedding_generator, 'embed_query'):
            return self.embedding_generator.embed_query(text)
        elif hasattr(self.embedding_generator, 'generate_embedding'):
            return await self.embedding_generator.generate_embedding(text)
        else:
            raise ValueError("Invalid embedding generator")
    
    def _compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity."""
        import numpy as np
        
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            
            # Remove from user index
            if memory.user_id and memory.user_id in self.user_memories:
                self.user_memories[memory.user_id] = [
                    mid for mid in self.user_memories[memory.user_id]
                    if mid != memory_id
                ]
            
            # Remove from embedding index
            if memory_id in self.embedding_index:
                del self.embedding_index[memory_id]
            
            del self.memories[memory_id]
            return True
        
        return False
    
    def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a user."""
        if user_id not in self.user_memories:
            return 0
        
        count = 0
        for memory_id in self.user_memories[user_id]:
            if memory_id in self.memories:
                del self.memories[memory_id]
                count += 1
            if memory_id in self.embedding_index:
                del self.embedding_index[memory_id]
        
        del self.user_memories[user_id]
        
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        by_type = {}
        for memory in self.memories.values():
            mt = memory.memory_type.value
            by_type[mt] = by_type.get(mt, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "total_users": len(self.user_memories),
            "total_profiles": len(self.user_profiles),
            "by_type": by_type,
            "embeddings_indexed": len(self.embedding_index)
        }
