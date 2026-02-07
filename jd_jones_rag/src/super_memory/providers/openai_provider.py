"""
OpenAI Memory Provider
Synchronizes memories with OpenAI's conversation memory system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx

from src.super_memory.providers.base_provider import (
    BaseMemoryProvider,
    ProviderMemory,
    ProviderStatus
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


class OpenAIMemoryProvider(BaseMemoryProvider):
    """
    Provider for syncing with OpenAI's memory system.
    Integrates with ChatGPT's memory feature via API.
    """
    
    BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            config: Additional configuration
        """
        super().__init__(
            provider_name="openai",
            api_key=api_key or settings.openai_api_key,
            config=config or {}
        )
        
        self.client: Optional[httpx.AsyncClient] = None
        self.organization_id = config.get("organization_id") if config else None
    
    async def connect(self) -> bool:
        """Establish connection to OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.organization_id:
                headers["OpenAI-Organization"] = self.organization_id
            
            self.client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=30.0
            )
            
            # Verify connection
            response = await self.client.get("/models")
            if response.status_code == 200:
                self.status = ProviderStatus.CONNECTED
                return True
            elif response.status_code == 401:
                self.status = ProviderStatus.UNAUTHORIZED
                return False
            else:
                self.status = ProviderStatus.ERROR
                return False
                
        except Exception as e:
            self.last_error = str(e)
            self.status = ProviderStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from OpenAI API."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.status = ProviderStatus.DISCONNECTED
    
    async def is_connected(self) -> bool:
        """Check if connected."""
        return self.client is not None and self.status == ProviderStatus.CONNECTED
    
    async def pull_memories(
        self,
        user_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ProviderMemory]:
        """
        Pull memories from OpenAI's memory system.
        
        Note: OpenAI's memory API structure is not publicly documented.
        This is a placeholder implementation.
        """
        if not await self.is_connected():
            await self.connect()
        
        memories = []
        
        try:
            # Placeholder: Actual implementation would use OpenAI's memory API
            # The actual endpoint and data structure may differ
            logger.debug(f"Pulling memories for user_id={user_id} (placeholder implementation)")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self.status = ProviderStatus.RATE_LIMITED
            elif e.response.status_code == 401:
                self.status = ProviderStatus.UNAUTHORIZED
            self.last_error = str(e)
        except Exception as e:
            self.last_error = str(e)
            self.status = ProviderStatus.ERROR
        
        return memories
    
    async def push_memories(
        self,
        user_id: str,
        memories: List[Dict[str, Any]]
    ) -> int:
        """Push memories to OpenAI's memory system."""
        if not await self.is_connected():
            await self.connect()
        
        pushed_count = 0
        
        try:
            for memory in memories:
                transformed = self.transform_to_provider_format(memory)
                # Placeholder: Actual implementation
                pushed_count += 1
                
        except Exception as e:
            self.last_error = str(e)
            self.status = ProviderStatus.ERROR
        
        return pushed_count
    
    async def delete_memory(
        self,
        user_id: str,
        memory_id: str
    ) -> bool:
        """Delete a memory from OpenAI's system."""
        if not await self.is_connected():
            await self.connect()
        
        try:
            # Placeholder: Actual implementation
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    async def get_memory_count(self, user_id: str) -> int:
        """Get total memory count for user."""
        if not await self.is_connected():
            await self.connect()
        
        try:
            # Placeholder: Actual implementation
            return 0
        except Exception as e:
            logger.warning(f"Failed to get memory count for user_id={user_id}: {e}")
            return 0
    
    def get_supported_memory_types(self) -> List[str]:
        """Get supported memory types."""
        return ["fact", "preference", "context"]
    
    def transform_to_provider_format(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Transform to OpenAI memory format."""
        return {
            "memory": memory.get("content", ""),
            "memory_type": self._map_memory_type(memory.get("memory_type", "fact")),
            "metadata": {
                "source": "jd_jones_rag",
                "category": memory.get("category"),
                "importance": memory.get("importance_score", 0.5)
            }
        }
    
    def _map_memory_type(self, memory_type: str) -> str:
        """Map internal memory types to OpenAI types."""
        mapping = {
            "fact": "fact",
            "preference": "preference",
            "context": "context",
            "instruction": "preference",
            "entity": "fact"
        }
        return mapping.get(memory_type, "fact")
    
    def transform_from_provider_format(self, provider_memory: ProviderMemory) -> Dict[str, Any]:
        """Transform from OpenAI memory format."""
        return {
            "content": provider_memory.content,
            "memory_type": provider_memory.memory_type,
            "source_provider": self.provider_name,
            "source_id": provider_memory.provider_id,
            "created_at": provider_memory.created_at,
            "metadata": provider_memory.metadata
        }
