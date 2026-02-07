"""
Memory Learner
Automatically extracts and stores learnable information from conversations.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.super_memory.memory_manager import SuperMemoryManager, Memory, MemoryType


class MemoryLearner:
    """
    Learns new information from user interactions automatically.
    Extracts facts, preferences, and context from conversations.
    """
    
    EXTRACTION_PROMPT = """You are an AI assistant that extracts learnable information from conversations.

Analyze the following interaction and extract any new information worth remembering about the user.

Categories of information to extract:
- FACTS: Factual information about the user (name, job, projects, etc.)
- PREFERENCES: User preferences (communication style, interests, etc.)
- CONTEXT: Current context (what they're working on, deadlines, etc.)
- INSTRUCTIONS: How the user wants to be helped (always/never do X)
- ENTITIES: Important people, projects, or things mentioned

For each piece of information:
1. Determine if it's worth remembering (general chitchat is NOT worth remembering)
2. Assign an importance score (0.0-1.0)
3. Assign a confidence score (0.0-1.0)

Respond in JSON format:
{
    "learned_items": [
        {
            "content": "The extracted information",
            "type": "fact|preference|context|instruction|entity",
            "category": "category name",
            "importance": 0.8,
            "confidence": 0.9,
            "reasoning": "Why this is worth remembering"
        }
    ],
    "suggested_preference_updates": {
        "field_name": "new_value"
    }
}

If there's nothing worth learning, return:
{"learned_items": [], "suggested_preference_updates": {}}

USER QUERY: {query}

ASSISTANT RESPONSE: {response}

EXISTING CONTEXT USED: {context_summary}
"""

    SUMMARY_PROMPT = """Summarize the following conversation between a user and an AI assistant.

Focus on:
1. Main topics discussed
2. Key entities mentioned (people, projects, products)
3. Any action items or next steps
4. Important decisions or conclusions

Keep the summary concise but comprehensive.

CONVERSATION:
{messages}

Respond in JSON format:
{
    "summary": "Brief summary of the conversation",
    "key_topics": ["topic1", "topic2"],
    "key_entities": {"type": "name"},
    "action_items": ["item1", "item2"],
    "sentiment": "positive|neutral|negative"
}
"""
    
    def __init__(self):
        """Initialize memory learner."""
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0)
        self.memory_manager = SuperMemoryManager()
    
    async def learn_from_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        context_used: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract and store learnable information from an interaction.
        
        Args:
            user_id: User identifier
            query: User's query
            response: Assistant's response
            session_id: Session identifier
            context_used: Summary of context used
            
        Returns:
            Learning results
        """
        # Build extraction prompt
        prompt = self.EXTRACTION_PROMPT.format(
            query=query,
            response=response,
            context_summary=context_used or "None"
        )
        
        try:
            # Call LLM for extraction
            messages = [
                SystemMessage(content="You extract learnable information from conversations."),
                HumanMessage(content=prompt)
            ]
            
            result = await self.llm.ainvoke(messages)
            
            # Parse response
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            extracted = json.loads(content.strip())
            
        except Exception as e:
            logger.error(f"Learning extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "memories_created": 0
            }
        
        # Process learned items
        memories_created = 0
        learned_items = extracted.get("learned_items", [])
        
        for item in learned_items:
            # Filter by confidence and importance
            if item.get("confidence", 0) < 0.7 or item.get("importance", 0) < 0.5:
                continue
            
            # Create memory
            try:
                memory_type = MemoryType(item.get("type", "fact"))
            except ValueError:
                memory_type = MemoryType.FACT
            
            memory = Memory(
                content=item["content"],
                memory_type=memory_type,
                category=item.get("category"),
                importance_score=item.get("importance", 0.5),
                confidence_score=item.get("confidence", 0.8),
                source_provider="system",
                metadata={
                    "extracted_from_session": session_id,
                    "reasoning": item.get("reasoning", ""),
                    "extracted_at": datetime.now().isoformat()
                }
            )
            
            await self.memory_manager.store_memory(user_id, memory)
            memories_created += 1
        
        # Process preference updates
        pref_updates = extracted.get("suggested_preference_updates", {})
        if pref_updates:
            await self.memory_manager.update_user_preferences(user_id, pref_updates)
        
        return {
            "success": True,
            "memories_created": memories_created,
            "learned_items": learned_items,
            "preference_updates": pref_updates
        }
    
    async def summarize_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Summarize a conversation and store the context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            messages: List of conversation messages
            
        Returns:
            Context ID if stored, None otherwise
        """
        if len(messages) < 4:  # Skip very short conversations
            return None
        
        # Format messages for summarization
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        
        prompt = self.SUMMARY_PROMPT.format(
            messages="\n".join(formatted)
        )
        
        try:
            messages_for_llm = [
                SystemMessage(content="You summarize conversations."),
                HumanMessage(content=prompt)
            ]
            
            result = await self.llm.ainvoke(messages_for_llm)
            
            # Parse response
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            summary_data = json.loads(content.strip())
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return None
        
        # Store conversation context
        context_id = await self.memory_manager.store_conversation_context(
            user_id=user_id,
            session_id=session_id,
            summary=summary_data.get("summary", ""),
            key_topics=summary_data.get("key_topics", []),
            key_entities=summary_data.get("key_entities", {}),
            action_items=summary_data.get("action_items", []),
            message_count=len(messages)
        )
        
        return context_id
    
    async def batch_learn(
        self,
        user_id: str,
        interactions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Learn from multiple interactions in batch.
        
        Args:
            user_id: User identifier
            interactions: List of {query, response} dicts
            
        Returns:
            Batch learning results
        """
        total_memories = 0
        results = []
        
        for interaction in interactions:
            result = await self.learn_from_interaction(
                user_id=user_id,
                query=interaction.get("query", ""),
                response=interaction.get("response", "")
            )
            
            total_memories += result.get("memories_created", 0)
            results.append(result)
        
        return {
            "interactions_processed": len(interactions),
            "total_memories_created": total_memories,
            "results": results
        }
    
    async def extract_entities(
        self,
        text: str
    ) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types to entity names
        """
        prompt = f"""Extract named entities from the following text.

Categories:
- PERSON: Names of people
- ORGANIZATION: Company/team names
- PROJECT: Project names
- PRODUCT: Product names
- LOCATION: Places
- DATE: Dates and times

Text: {text}

Respond in JSON: {{"PERSON": [], "ORGANIZATION": [], "PROJECT": [], "PRODUCT": [], "LOCATION": [], "DATE": []}}
"""
        
        try:
            messages = [HumanMessage(content=prompt)]
            result = await self.llm.ainvoke(messages)
            
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            return json.loads(content.strip())
            
        except Exception:
            return {}
