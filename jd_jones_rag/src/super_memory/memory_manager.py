"""
Super Memory Manager
Core component for memory storage, retrieval, and management.
Uses PostgreSQL with pgvector for vector similarity search.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

import asyncpg
from pgvector.asyncpg import register_vector
import numpy as np
import redis.asyncio as redis

from src.config.settings import settings
from src.data_ingestion.embedding_generator import EmbeddingGenerator


class MemoryType(str, Enum):
    """Types of memories."""
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    INSTRUCTION = "instruction"
    ENTITY = "entity"


@dataclass
class Memory:
    """Memory data structure."""
    content: str
    memory_type: MemoryType
    category: Optional[str] = None
    importance_score: float = 0.5
    confidence_score: float = 0.8
    tags: List[str] = field(default_factory=list)
    source_provider: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Set after storage
    memory_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, MemoryType) else self.memory_type,
            "category": self.category,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "tags": self.tags,
            "source_provider": self.source_provider,
            "source_id": self.source_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class UserPreferences:
    """User preferences data structure."""
    user_id: str
    preferred_response_style: str = "balanced"
    preferred_tone: str = "professional"
    preferred_format: Optional[str] = None
    primary_role: Optional[str] = None
    primary_department: Optional[str] = None
    expertise_areas: List[str] = field(default_factory=list)
    current_projects: Dict[str, Any] = field(default_factory=dict)
    frequent_topics: List[str] = field(default_factory=list)
    technical_level: str = "intermediate"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "preferred_response_style": self.preferred_response_style,
            "preferred_tone": self.preferred_tone,
            "preferred_format": self.preferred_format,
            "primary_role": self.primary_role,
            "primary_department": self.primary_department,
            "expertise_areas": self.expertise_areas,
            "current_projects": self.current_projects,
            "frequent_topics": self.frequent_topics,
            "technical_level": self.technical_level
        }


class SuperMemoryManager:
    """
    Manages memory storage and retrieval using PostgreSQL + pgvector.
    Provides semantic search, caching, and batch operations.
    """
    
    def __init__(self):
        """Initialize Super Memory Manager."""
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[redis.Redis] = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False
    
    async def initialize(self):
        """Initialize database and Redis connections."""
        if self._initialized:
            return
        
        # Create PostgreSQL connection pool
        db_url = settings.database_url.replace("+asyncpg", "").replace("postgresql", "postgres")
        
        self.pool = await asyncpg.create_pool(
            db_url,
            min_size=5,
            max_size=20,
            init=self._init_connection
        )
        
        # Create Redis connection
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        
        self._initialized = True
    
    async def _init_connection(self, conn: asyncpg.Connection):
        """Initialize pgvector on each connection."""
        await register_vector(conn)
    
    async def close(self):
        """Close connections."""
        if self.pool:
            await self.pool.close()
        if self.redis:
            await self.redis.close()
        self._initialized = False
    
    async def store_memory(
        self,
        user_id: str,
        memory: Memory
    ) -> Memory:
        """
        Store a single memory with embedding.
        
        Args:
            user_id: User identifier
            memory: Memory to store
            
        Returns:
            Memory with ID and timestamps
        """
        await self.initialize()
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(memory.content)
        
        # Generate memory ID
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_memories (
                    memory_id, user_id, content, summary, memory_type,
                    category, tags, source_provider, source_id,
                    importance_score, confidence_score, embedding, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    importance_score = EXCLUDED.importance_score,
                    confidence_score = EXCLUDED.confidence_score,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            """,
                memory_id,
                user_id,
                memory.content,
                memory.content[:200],  # Summary
                memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
                memory.category,
                memory.tags,
                memory.source_provider,
                memory.source_id,
                memory.importance_score,
                memory.confidence_score,
                np.array(embedding),
                json.dumps(memory.metadata)
            )
        
        # Update memory object
        memory.memory_id = memory_id
        memory.embedding = embedding
        memory.created_at = datetime.now()
        
        # Invalidate cache
        await self._invalidate_user_cache(user_id)
        
        return memory
    
    async def store_memories_batch(
        self,
        user_id: str,
        memories: List[Memory]
    ) -> List[Memory]:
        """
        Store multiple memories in batch.
        
        Args:
            user_id: User identifier
            memories: List of memories to store
            
        Returns:
            List of memories with IDs
        """
        await self.initialize()
        
        if not memories:
            return []
        
        # Generate embeddings in batch
        contents = [m.content for m in memories]
        embeddings = self.embedding_generator.generate_embeddings_batch(contents)
        
        stored_memories = []
        
        async with self.pool.acquire() as conn:
            for memory, embedding in zip(memories, embeddings):
                memory_id = f"mem_{uuid.uuid4().hex[:12]}"
                
                await conn.execute("""
                    INSERT INTO user_memories (
                        memory_id, user_id, content, summary, memory_type,
                        category, tags, source_provider, source_id,
                        importance_score, confidence_score, embedding, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    memory_id,
                    user_id,
                    memory.content,
                    memory.content[:200],
                    memory.memory_type.value if isinstance(memory.memory_type, MemoryType) else memory.memory_type,
                    memory.category,
                    memory.tags,
                    memory.source_provider,
                    memory.source_id,
                    memory.importance_score,
                    memory.confidence_score,
                    np.array(embedding),
                    json.dumps(memory.metadata)
                )
                
                memory.memory_id = memory_id
                memory.embedding = embedding
                memory.created_at = datetime.now()
                stored_memories.append(memory)
        
        # Invalidate cache
        await self._invalidate_user_cache(user_id)
        
        return stored_memories
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search memories using semantic similarity.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
            memory_type: Filter by type
            min_importance: Minimum importance score
            min_confidence: Minimum confidence score
            
        Returns:
            List of memories with relevance scores
        """
        await self.initialize()
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        async with self.pool.acquire() as conn:
            # Build query with filters
            type_filter = f"AND memory_type = '{memory_type.value}'" if memory_type else ""
            
            rows = await conn.fetch(f"""
                SELECT 
                    memory_id, content, memory_type, category, tags,
                    importance_score, confidence_score, source_provider,
                    created_at, last_accessed_at,
                    1 - (embedding <=> $1) as relevance_score
                FROM user_memories
                WHERE user_id = $2
                    AND importance_score >= $3
                    AND confidence_score >= $4
                    {type_filter}
                ORDER BY embedding <=> $1
                LIMIT $5
            """,
                np.array(query_embedding),
                user_id,
                min_importance,
                min_confidence,
                limit
            )
            
            results = []
            for row in rows:
                # Update access time
                await conn.execute("""
                    UPDATE user_memories 
                    SET last_accessed_at = NOW(), access_count = access_count + 1
                    WHERE memory_id = $1
                """, row['memory_id'])
                
                results.append({
                    "memory_id": row['memory_id'],
                    "content": row['content'],
                    "memory_type": row['memory_type'],
                    "category": row['category'],
                    "tags": row['tags'],
                    "importance_score": float(row['importance_score']),
                    "confidence_score": float(row['confidence_score']),
                    "source_provider": row['source_provider'],
                    "relevance_score": float(row['relevance_score']),
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None
                })
            
            return results
    
    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get user memories with optional filtering.
        
        Args:
            user_id: User identifier
            memory_type: Filter by type
            limit: Maximum results
            
        Returns:
            List of memories
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            type_filter = f"AND memory_type = '{memory_type.value}'" if memory_type else ""
            
            rows = await conn.fetch(f"""
                SELECT 
                    memory_id, content, memory_type, category, tags,
                    importance_score, confidence_score, source_provider,
                    created_at, last_accessed_at
                FROM user_memories
                WHERE user_id = $1 {type_filter}
                ORDER BY importance_score DESC, created_at DESC
                LIMIT $2
            """, user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_recent_context(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recently accessed or created memories.
        
        Args:
            user_id: User identifier
            hours: Time window in hours
            limit: Maximum results
            
        Returns:
            List of recent memories
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    memory_id, content, memory_type, category, tags,
                    importance_score, confidence_score,
                    created_at, last_accessed_at
                FROM user_memories
                WHERE user_id = $1
                    AND (created_at > NOW() - INTERVAL '%s hours'
                         OR last_accessed_at > NOW() - INTERVAL '%s hours')
                ORDER BY COALESCE(last_accessed_at, created_at) DESC
                LIMIT $2
            """ % (hours, hours), user_id, limit)
            
            return [dict(row) for row in rows]
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferences or None
        """
        await self.initialize()
        
        # Check cache first
        cache_key = f"user_prefs:{user_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return UserPreferences(**data)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM user_preferences WHERE user_id = $1
            """, user_id)
            
            if not row:
                return None
            
            prefs = UserPreferences(
                user_id=row['user_id'],
                preferred_response_style=row['preferred_response_style'],
                preferred_tone=row['preferred_tone'],
                preferred_format=row['preferred_format'],
                primary_role=row['primary_role'],
                primary_department=row['primary_department'],
                expertise_areas=row['expertise_areas'] or [],
                current_projects=row['current_projects'] or {},
                frequent_topics=row['frequent_topics'] or [],
                technical_level=row['technical_level']
            )
            
            # Cache for 1 hour
            await self.redis.setex(
                cache_key,
                settings.memory_cache_ttl,
                json.dumps(prefs.to_dict())
            )
            
            return prefs
    
    async def update_user_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> UserPreferences:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            updates: Dictionary of updates
            
        Returns:
            Updated UserPreferences
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            # Upsert preferences
            await conn.execute("""
                INSERT INTO user_preferences (
                    user_id, preferred_response_style, preferred_tone,
                    preferred_format, primary_role, primary_department,
                    expertise_areas, current_projects, frequent_topics, technical_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (user_id) DO UPDATE SET
                    preferred_response_style = COALESCE(EXCLUDED.preferred_response_style, user_preferences.preferred_response_style),
                    preferred_tone = COALESCE(EXCLUDED.preferred_tone, user_preferences.preferred_tone),
                    preferred_format = COALESCE(EXCLUDED.preferred_format, user_preferences.preferred_format),
                    primary_role = COALESCE(EXCLUDED.primary_role, user_preferences.primary_role),
                    primary_department = COALESCE(EXCLUDED.primary_department, user_preferences.primary_department),
                    expertise_areas = COALESCE(EXCLUDED.expertise_areas, user_preferences.expertise_areas),
                    current_projects = COALESCE(EXCLUDED.current_projects, user_preferences.current_projects),
                    frequent_topics = COALESCE(EXCLUDED.frequent_topics, user_preferences.frequent_topics),
                    technical_level = COALESCE(EXCLUDED.technical_level, user_preferences.technical_level),
                    updated_at = NOW()
            """,
                user_id,
                updates.get('preferred_response_style', 'balanced'),
                updates.get('preferred_tone', 'professional'),
                updates.get('preferred_format'),
                updates.get('primary_role'),
                updates.get('primary_department'),
                updates.get('expertise_areas', []),
                json.dumps(updates.get('current_projects', {})),
                updates.get('frequent_topics', []),
                updates.get('technical_level', 'intermediate')
            )
        
        # Invalidate cache
        await self._invalidate_user_cache(user_id)
        
        return await self.get_user_preferences(user_id)
    
    async def store_conversation_context(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        key_topics: List[str],
        key_entities: Dict[str, Any],
        action_items: List[str],
        message_count: int
    ) -> str:
        """
        Store conversation context summary.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            summary: Conversation summary
            key_topics: List of key topics
            key_entities: Extracted entities
            action_items: Action items from conversation
            message_count: Number of messages
            
        Returns:
            Context ID
        """
        await self.initialize()
        
        context_id = f"ctx_{uuid.uuid4().hex[:12]}"
        
        # Generate embedding for the summary
        embedding = self.embedding_generator.generate_embedding(summary)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversation_contexts (
                    context_id, user_id, session_id, summary,
                    key_topics, key_entities, action_items,
                    message_count, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                context_id,
                user_id,
                session_id,
                summary,
                key_topics,
                json.dumps(key_entities),
                action_items,
                message_count,
                np.array(embedding)
            )
        
        return context_id
    
    async def get_relevant_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant past conversations using semantic search.
        
        Args:
            user_id: User identifier
            query: Query to find relevant conversations
            limit: Maximum results
            
        Returns:
            List of relevant conversation contexts
        """
        await self.initialize()
        
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    context_id, session_id, summary, key_topics,
                    key_entities, action_items, message_count,
                    created_at,
                    1 - (embedding <=> $1) as relevance_score
                FROM conversation_contexts
                WHERE user_id = $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """,
                np.array(query_embedding),
                user_id,
                limit
            )
            
            return [dict(row) for row in rows]
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: Memory identifier
            user_id: User identifier (for authorization)
            
        Returns:
            True if deleted
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM user_memories
                WHERE memory_id = $1 AND user_id = $2
            """, memory_id, user_id)
            
            deleted = result.split()[-1] != '0'
            
            if deleted:
                await self._invalidate_user_cache(user_id)
            
            return deleted
    
    async def _invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        if self.redis:
            keys = await self.redis.keys(f"*:{user_id}*")
            if keys:
                await self.redis.delete(*keys)
    
    async def get_auto_sync_users(self) -> List[str]:
        """
        Get list of users with auto-sync enabled.
        
        Returns:
            List of user IDs
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT user_id FROM user_preferences
                WHERE (metadata->>'auto_sync_enabled')::boolean = true
                OR NOT EXISTS (
                    SELECT 1 FROM user_preferences up2 
                    WHERE up2.user_id = user_preferences.user_id 
                    AND up2.metadata ? 'auto_sync_enabled'
                )
            """)
            return [row['user_id'] for row in rows]
    
    async def cleanup_old_sync_logs(self, cutoff_date: datetime) -> int:
        """
        Clean up old sync history records.
        
        Args:
            cutoff_date: Delete records older than this date
            
        Returns:
            Number of records deleted
        """
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM sync_history
                WHERE started_at < $1
            """, cutoff_date)
            
            # Parse "DELETE n" result
            count = int(result.split()[-1])
            return count
    
    async def refresh_all_embeddings(self, batch_size: int = 100) -> int:
        """
        Refresh embeddings for all memories.
        Useful when embedding model is updated.
        
        Args:
            batch_size: Number of memories to process at once
            
        Returns:
            Total number of embeddings refreshed
        """
        await self.initialize()
        
        total_refreshed = 0
        offset = 0
        
        while True:
            async with self.pool.acquire() as conn:
                # Get batch of memories
                rows = await conn.fetch("""
                    SELECT memory_id, content FROM user_memories
                    ORDER BY memory_id
                    LIMIT $1 OFFSET $2
                """, batch_size, offset)
                
                if not rows:
                    break
                
                # Generate new embeddings
                contents = [row['content'] for row in rows]
                embeddings = self.embedding_generator.generate_embeddings_batch(contents)
                
                # Update embeddings
                for row, embedding in zip(rows, embeddings):
                    await conn.execute("""
                        UPDATE user_memories
                        SET embedding = $1, updated_at = NOW()
                        WHERE memory_id = $2
                    """, np.array(embedding), row['memory_id'])
                
                total_refreshed += len(rows)
                offset += batch_size
        
        return total_refreshed
    
    async def summarize_and_store_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Summarize a conversation and store it.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            messages: List of conversation messages
            
        Returns:
            Stored context data
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Generate summary using LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        conversation_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        ])
        
        summary_prompt = f"""Summarize this conversation in 2-3 sentences, 
        noting the main topics discussed and any action items:
        
        {conversation_text}"""
        
        response = await llm.ainvoke([
            SystemMessage(content="You are a conversation summarizer."),
            HumanMessage(content=summary_prompt)
        ])
        
        summary = response.content
        
        # Extract topics (simple approach)
        topics = [word.strip() for word in summary.split() 
                  if len(word) > 5][:10]
        
        # Store the context
        context_id = f"ctx_{uuid.uuid4().hex[:12]}"
        embedding = self.embedding_generator.generate_embedding(summary)
        
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversation_contexts (
                    context_id, user_id, session_id, summary,
                    key_topics, message_count, embedding, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            """,
                context_id,
                user_id,
                session_id,
                summary,
                topics,
                len(messages),
                np.array(embedding)
            )
        
        return {
            "context_id": context_id,
            "summary": summary,
            "topics": topics,
            "message_count": len(messages)
        }
