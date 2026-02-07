"""
Runtime Context Loader
Loads and assembles user context at query time for personalized responses.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from src.super_memory.memory_manager import SuperMemoryManager, MemoryType
from src.knowledge_base.retriever import HierarchicalRetriever, UserRole
from src.auth.authentication import User


@dataclass
class LoadedContext:
    """Assembled context for LLM."""
    user_profile: Dict[str, Any]
    memories: List[Dict[str, Any]]
    conversations: List[Dict[str, Any]]
    rag_documents: List[Dict[str, Any]]
    context_string: str
    metadata: Dict[str, Any]


class RuntimeContextLoader:
    """
    Loads user context at runtime for query processing.
    Combines memories, preferences, conversations, and RAG documents.
    """
    
    def __init__(self):
        """Initialize context loader."""
        self.memory_manager = SuperMemoryManager()
        self.retriever = HierarchicalRetriever()
    
    async def load_context_for_query(
        self,
        user: User,
        query: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_memories: int = 10,
        max_conversations: int = 3,
        max_rag_docs: int = 6
    ) -> LoadedContext:
        """
        Load all relevant context for a query.
        
        Args:
            user: Authenticated user
            query: User's query
            session_id: Current session ID
            conversation_history: Current conversation
            max_memories: Maximum memories to load
            max_conversations: Maximum past conversations
            max_rag_docs: Maximum RAG documents
            
        Returns:
            LoadedContext with assembled information
        """
        # Run all context loading in parallel
        results = await asyncio.gather(
            self._load_user_context(user.user_id),
            self._load_memory_context(user.user_id, query, max_memories),
            self._load_conversation_context(user.user_id, query, max_conversations),
            self._load_rag_context(user, query, max_rag_docs),
            return_exceptions=True
        )
        
        # Unpack results
        user_context = results[0] if not isinstance(results[0], Exception) else {}
        memory_context = results[1] if not isinstance(results[1], Exception) else []
        conversation_context = results[2] if not isinstance(results[2], Exception) else []
        rag_context = results[3] if not isinstance(results[3], Exception) else []
        
        # Assemble context string
        context_string = self._assemble_context_string(
            user_context=user_context,
            memories=memory_context,
            conversations=conversation_context,
            rag_documents=rag_context,
            user=user
        )
        
        # Build metadata
        metadata = {
            "memories_loaded": len(memory_context),
            "conversations_loaded": len(conversation_context),
            "rag_docs_loaded": len(rag_context),
            "total_context_chars": len(context_string),
            "loaded_at": datetime.now().isoformat()
        }
        
        return LoadedContext(
            user_profile=user_context,
            memories=memory_context,
            conversations=conversation_context,
            rag_documents=rag_context,
            context_string=context_string,
            metadata=metadata
        )
    
    async def _load_user_context(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences and profile."""
        prefs = await self.memory_manager.get_user_preferences(user_id)
        
        if prefs:
            return prefs.to_dict()
        return {}
    
    async def _load_memory_context(
        self,
        user_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Load relevant memories via semantic search."""
        # Get semantically relevant memories
        semantic_memories = await self.memory_manager.search_memories(
            user_id=user_id,
            query=query,
            limit=limit,
            min_importance=0.3,
            min_confidence=0.5
        )
        
        # Also get high-importance memories
        important_memories = await self.memory_manager.get_user_memories(
            user_id=user_id,
            limit=5
        )
        
        # Merge and deduplicate
        seen_ids = set()
        merged = []
        
        for mem in semantic_memories:
            if mem['memory_id'] not in seen_ids:
                seen_ids.add(mem['memory_id'])
                merged.append(mem)
        
        for mem in important_memories:
            if mem['memory_id'] not in seen_ids:
                seen_ids.add(mem['memory_id'])
                merged.append(mem)
        
        return merged[:limit]
    
    async def _load_conversation_context(
        self,
        user_id: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Load relevant past conversations."""
        return await self.memory_manager.get_relevant_conversations(
            user_id=user_id,
            query=query,
            limit=limit
        )
    
    async def _load_rag_context(
        self,
        user: User,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Load RAG documents based on user access."""
        try:
            user_role = UserRole(user.role)
        except ValueError:
            user_role = UserRole.EMPLOYEE
        
        retrieval_response = self.retriever.retrieve(
            query=query,
            user_role=user_role,
            user_department=user.department,
            n_results=limit
        )
        
        return [r.to_dict() for r in retrieval_response.all_results]
    
    def _assemble_context_string(
        self,
        user_context: Dict[str, Any],
        memories: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]],
        rag_documents: List[Dict[str, Any]],
        user: User
    ) -> str:
        """Assemble all context into a formatted string for LLM."""
        sections = []
        
        # User Profile Section
        profile_parts = [
            f"Name: {user.full_name}",
            f"Role: {user.role}",
            f"Department: {user.department or 'General'}"
        ]
        
        if user_context:
            if user_context.get('preferred_response_style'):
                profile_parts.append(f"Prefers: {user_context['preferred_response_style']} responses")
            if user_context.get('technical_level'):
                profile_parts.append(f"Technical level: {user_context['technical_level']}")
            if user_context.get('expertise_areas'):
                profile_parts.append(f"Expertise: {', '.join(user_context['expertise_areas'][:3])}")
        
        sections.append("=== USER PROFILE ===\n" + "\n".join(profile_parts))
        
        # Memories Section
        if memories:
            memory_parts = []
            for mem in memories[:10]:
                mem_type = mem.get('memory_type', 'info')
                content = mem.get('content', '')[:200]
                memory_parts.append(f"• [{mem_type}] {content}")
            
            sections.append(
                "=== WHAT I KNOW ABOUT THIS USER ===\n" + 
                "\n".join(memory_parts)
            )
        
        # Relevant Conversations Section
        if conversations:
            conv_parts = []
            for conv in conversations[:3]:
                summary = conv.get('summary', '')[:150]
                topics = conv.get('key_topics', [])[:3]
                topics_str = ", ".join(topics) if topics else "general"
                conv_parts.append(f"• Topics: {topics_str}\n  Summary: {summary}")
            
            sections.append(
                "=== RELEVANT PAST CONVERSATIONS ===\n" +
                "\n".join(conv_parts)
            )
        
        # RAG Documents Section
        if rag_documents:
            doc_parts = []
            for doc in rag_documents[:6]:
                source = doc.get('metadata', {}).get('file_name', 'Knowledge Base')
                content = doc.get('content', '')[:300]
                doc_parts.append(f"[Source: {source}]\n{content}")
            
            sections.append(
                "=== RELEVANT COMPANY KNOWLEDGE ===\n" +
                "\n---\n".join(doc_parts)
            )
        
        return "\n\n".join(sections)
    
    async def get_quick_context(
        self,
        user_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Get quick context for simple queries.
        Lighter-weight than full context loading.
        
        Args:
            user_id: User identifier
            query: User's query
            
        Returns:
            Quick context dict
        """
        # Just get relevant memories and preferences
        results = await asyncio.gather(
            self.memory_manager.get_user_preferences(user_id),
            self.memory_manager.search_memories(
                user_id=user_id,
                query=query,
                limit=5
            ),
            return_exceptions=True
        )
        
        prefs = results[0] if not isinstance(results[0], Exception) else None
        memories = results[1] if not isinstance(results[1], Exception) else []
        
        return {
            "preferences": prefs.to_dict() if prefs else {},
            "relevant_memories": memories,
            "context_summary": self._build_quick_summary(prefs, memories)
        }
    
    def _build_quick_summary(
        self,
        prefs: Optional[Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """Build a quick context summary."""
        parts = []
        
        if prefs:
            parts.append(f"User prefers {prefs.preferred_response_style} responses.")
        
        if memories:
            facts = [m['content'][:100] for m in memories if m.get('memory_type') == 'fact'][:3]
            if facts:
                parts.append("Known facts: " + "; ".join(facts))
        
        return " ".join(parts) if parts else "No specific context available."
