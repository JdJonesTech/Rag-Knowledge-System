"""
Base Memory Provider
Abstract base class for memory synchronization providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SyncDirection(str, Enum):
    """Synchronization direction."""
    PULL = "pull"  # From provider to local
    PUSH = "push"  # From local to provider
    BIDIRECTIONAL = "bidirectional"


class ProviderStatus(str, Enum):
    """Provider connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"


@dataclass
class ProviderMemory:
    """Memory structure from external provider."""
    provider_id: str
    content: str
    memory_type: str
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ProviderSyncResult:
    """Result of a sync operation."""
    provider_name: str
    success: bool
    memories_pulled: int = 0
    memories_pushed: int = 0
    memories_updated: int = 0
    memories_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    sync_started_at: Optional[datetime] = None
    sync_completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_name": self.provider_name,
            "success": self.success,
            "memories_pulled": self.memories_pulled,
            "memories_pushed": self.memories_pushed,
            "memories_updated": self.memories_updated,
            "memories_skipped": self.memories_skipped,
            "errors": self.errors,
            "sync_started_at": self.sync_started_at.isoformat() if self.sync_started_at else None,
            "sync_completed_at": self.sync_completed_at.isoformat() if self.sync_completed_at else None,
            "duration_seconds": (
                (self.sync_completed_at - self.sync_started_at).total_seconds()
                if self.sync_started_at and self.sync_completed_at else None
            )
        }


class BaseMemoryProvider(ABC):
    """
    Abstract base class for memory providers.
    All provider implementations must inherit from this class.
    """
    
    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provider.
        
        Args:
            provider_name: Unique provider name
            api_key: API key for authentication
            config: Additional configuration
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.config = config or {}
        self.status = ProviderStatus.DISCONNECTED
        self.last_sync: Optional[datetime] = None
        self.last_error: Optional[str] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the provider.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the provider."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if provider is connected.
        
        Returns:
            True if connected
        """
        pass
    
    @abstractmethod
    async def pull_memories(
        self,
        user_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ProviderMemory]:
        """
        Pull memories from the provider.
        
        Args:
            user_id: User identifier
            since: Only pull memories after this time
            limit: Maximum memories to pull
            
        Returns:
            List of provider memories
        """
        pass
    
    @abstractmethod
    async def push_memories(
        self,
        user_id: str,
        memories: List[Dict[str, Any]]
    ) -> int:
        """
        Push memories to the provider.
        
        Args:
            user_id: User identifier
            memories: List of memory dicts to push
            
        Returns:
            Number of memories pushed
        """
        pass
    
    @abstractmethod
    async def delete_memory(
        self,
        user_id: str,
        memory_id: str
    ) -> bool:
        """
        Delete a memory from the provider.
        
        Args:
            user_id: User identifier
            memory_id: Memory identifier
            
        Returns:
            True if deleted
        """
        pass
    
    @abstractmethod
    async def get_memory_count(self, user_id: str) -> int:
        """
        Get total memory count for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory count
        """
        pass
    
    async def sync(
        self,
        user_id: str,
        local_memories: List[Dict[str, Any]],
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        since: Optional[datetime] = None
    ) -> ProviderSyncResult:
        """
        Synchronize memories with the provider.
        
        Args:
            user_id: User identifier
            local_memories: Local memories to sync
            direction: Sync direction
            since: Only sync memories after this time
            
        Returns:
            Sync result
        """
        result = ProviderSyncResult(
            provider_name=self.provider_name,
            success=True,
            sync_started_at=datetime.now()
        )
        
        try:
            # Ensure connection
            if not await self.is_connected():
                if not await self.connect():
                    result.success = False
                    result.errors.append("Failed to connect to provider")
                    return result
            
            # Pull memories
            if direction in [SyncDirection.PULL, SyncDirection.BIDIRECTIONAL]:
                pulled = await self.pull_memories(user_id, since=since)
                result.memories_pulled = len(pulled)
            
            # Push memories
            if direction in [SyncDirection.PUSH, SyncDirection.BIDIRECTIONAL]:
                pushed = await self.push_memories(user_id, local_memories)
                result.memories_pushed = pushed
            
            self.last_sync = datetime.now()
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            self.last_error = str(e)
            self.status = ProviderStatus.ERROR
        
        result.sync_completed_at = datetime.now()
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_error": self.last_error
        }
    
    @abstractmethod
    def get_supported_memory_types(self) -> List[str]:
        """
        Get memory types supported by this provider.
        
        Returns:
            List of supported memory type strings
        """
        pass
    
    def transform_to_provider_format(
        self,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform local memory to provider format.
        Override in subclasses for provider-specific formatting.
        
        Args:
            memory: Local memory dict
            
        Returns:
            Provider-formatted memory
        """
        return memory
    
    def transform_from_provider_format(
        self,
        provider_memory: ProviderMemory
    ) -> Dict[str, Any]:
        """
        Transform provider memory to local format.
        Override in subclasses for provider-specific parsing.
        
        Args:
            provider_memory: Provider memory
            
        Returns:
            Local memory dict
        """
        return {
            "content": provider_memory.content,
            "memory_type": provider_memory.memory_type,
            "source_provider": self.provider_name,
            "source_id": provider_memory.provider_id,
            "created_at": provider_memory.created_at,
            "metadata": provider_memory.metadata
        }
