"""
Conversation Memory
Manages short-term memory for conversation context.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json


@dataclass
class Message:
    """A single conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ConversationContext:
    """Context maintained during a conversation."""
    session_id: str
    user_id: Optional[str]
    messages: List[Message] = field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    collected_parameters: Dict[str, Any] = field(default_factory=dict)
    current_topic: Optional[str] = None
    topics_discussed: List[str] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "messages": [m.to_dict() for m in self.messages[-10:]],  # Last 10
            "extracted_entities": self.extracted_entities,
            "collected_parameters": self.collected_parameters,
            "current_topic": self.current_topic,
            "topics_discussed": self.topics_discussed,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


class ConversationMemory:
    """
    Manages short-term conversation memory.
    
    Features:
    - Message history with sliding window
    - Entity extraction and tracking
    - Parameter collection across turns
    - Topic tracking
    - Automatic summarization
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        context_window: int = 10,
        ttl_hours: int = 24
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum messages to store per session
            context_window: Number of recent messages for context
            ttl_hours: Time-to-live for inactive sessions
        """
        self.max_messages = max_messages
        self.context_window = context_window
        self.ttl_hours = ttl_hours
        
        # Active conversations
        self.conversations: Dict[str, ConversationContext] = {}
    
    def get_or_create(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing conversation or create new one."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id
            )
        
        return self.conversations[session_id]
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to conversation.
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            The created Message
        """
        context = self.get_or_create(session_id)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        context.messages.append(message)
        context.last_activity = datetime.now()
        
        # Trim if exceeds max
        if len(context.messages) > self.max_messages:
            # Summarize old messages before removing
            self._summarize_old_messages(context)
            context.messages = context.messages[-self.max_messages:]
        
        # Extract entities from user messages
        if role == "user":
            self._extract_entities(context, content)
        
        return message
    
    def get_context(
        self,
        session_id: str,
        num_messages: Optional[int] = None
    ) -> Optional[ConversationContext]:
        """
        Get conversation context.
        
        Args:
            session_id: Session identifier
            num_messages: Number of recent messages to include
            
        Returns:
            ConversationContext or None
        """
        if session_id not in self.conversations:
            return None
        
        context = self.conversations[session_id]
        
        # Return copy with limited messages
        if num_messages:
            limited = ConversationContext(
                session_id=context.session_id,
                user_id=context.user_id,
                messages=context.messages[-num_messages:],
                extracted_entities=context.extracted_entities,
                collected_parameters=context.collected_parameters,
                current_topic=context.current_topic,
                topics_discussed=context.topics_discussed,
                summary=context.summary,
                created_at=context.created_at,
                last_activity=context.last_activity
            )
            return limited
        
        return context
    
    def get_messages_for_llm(
        self,
        session_id: str,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM input.
        
        Args:
            session_id: Session identifier
            include_summary: Whether to include conversation summary
            
        Returns:
            List of message dictionaries
        """
        context = self.get_or_create(session_id)
        messages = []
        
        # Add summary as system message
        if include_summary and context.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {context.summary}"
            })
        
        # Add recent messages
        for msg in context.messages[-self.context_window:]:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def update_parameters(
        self,
        session_id: str,
        parameters: Dict[str, Any]
    ):
        """Update collected parameters."""
        context = self.get_or_create(session_id)
        context.collected_parameters.update(parameters)
        context.last_activity = datetime.now()
    
    def get_parameters(self, session_id: str) -> Dict[str, Any]:
        """Get collected parameters."""
        context = self.conversations.get(session_id)
        return context.collected_parameters if context else {}
    
    def set_topic(self, session_id: str, topic: str):
        """Set current conversation topic."""
        context = self.get_or_create(session_id)
        
        if context.current_topic and context.current_topic != topic:
            if context.current_topic not in context.topics_discussed:
                context.topics_discussed.append(context.current_topic)
        
        context.current_topic = topic
        context.last_activity = datetime.now()
    
    def _extract_entities(self, context: ConversationContext, content: str):
        """Extract named entities from content."""
        import re
        
        # Product patterns
        products = re.findall(r'\b(?:PACMAAN|FLEXSEAL|EXPANSOFLEX)[-\w]*\b', content, re.I)
        if products:
            if "products" not in context.extracted_entities:
                context.extracted_entities["products"] = []
            context.extracted_entities["products"].extend(products)
        
        # Standards
        standards = re.findall(r'\b(?:API\s*\d+|Shell\s*SPE|FDA|ASME|ASTM)\b', content, re.I)
        if standards:
            if "standards" not in context.extracted_entities:
                context.extracted_entities["standards"] = []
            context.extracted_entities["standards"].extend(standards)
        
        # Companies
        companies = re.findall(r'\b(?:Aramco|Shell|Reliance|ONGC)\b', content, re.I)
        if companies:
            if "companies" not in context.extracted_entities:
                context.extracted_entities["companies"] = []
            context.extracted_entities["companies"].extend(companies)
    
    def _summarize_old_messages(self, context: ConversationContext):
        """Summarize old messages before trimming."""
        if len(context.messages) <= self.max_messages:
            return
        
        # Get messages to summarize
        to_summarize = context.messages[:len(context.messages) - self.max_messages + 10]
        
        # Simple summary (in production, use LLM)
        user_messages = [m.content for m in to_summarize if m.role == "user"]
        summary_parts = []
        
        if user_messages:
            summary_parts.append(f"User discussed: {', '.join(user_messages[:3])}")
        
        if context.extracted_entities:
            for ent_type, entities in context.extracted_entities.items():
                if entities:
                    summary_parts.append(f"{ent_type}: {', '.join(set(entities[:3]))}")
        
        if summary_parts:
            new_summary = "; ".join(summary_parts)
            if context.summary:
                context.summary = f"{context.summary}. {new_summary}"
            else:
                context.summary = new_summary
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        
        expired = [
            sid for sid, ctx in self.conversations.items()
            if ctx.last_activity < cutoff
        ]
        
        for sid in expired:
            del self.conversations[sid]
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_messages = sum(len(c.messages) for c in self.conversations.values())
        
        return {
            "active_sessions": len(self.conversations),
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / len(self.conversations) if self.conversations else 0
        }
