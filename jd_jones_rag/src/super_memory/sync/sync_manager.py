"""
Sync Manager
Orchestrates memory synchronization across multiple providers.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.super_memory.memory_manager import SuperMemoryManager
from src.super_memory.providers.base_provider import (
    BaseMemoryProvider,
    ProviderSyncResult,
    SyncDirection
)
from src.super_memory.providers.claude_provider import ClaudeMemoryProvider
from src.super_memory.providers.openai_provider import OpenAIMemoryProvider
from src.super_memory.providers.gemini_provider import GeminiMemoryProvider
from src.super_memory.sync.conflict_resolver import ConflictResolver
from src.config.settings import settings


@dataclass
class SyncStatus:
    """Overall sync status."""
    sync_id: str
    user_id: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    provider_results: Dict[str, ProviderSyncResult] = field(default_factory=dict)
    total_memories_synced: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sync_id": self.sync_id,
            "user_id": self.user_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "provider_results": {
                k: v.to_dict() for k, v in self.provider_results.items()
            },
            "total_memories_synced": self.total_memories_synced,
            "errors": self.errors,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at else None
            )
        }


class SyncManager:
    """
    Manages synchronization across multiple memory providers.
    Handles conflict resolution and deduplication.
    """
    
    PROVIDER_CLASSES = {
        "claude": ClaudeMemoryProvider,
        "openai": OpenAIMemoryProvider,
        "gemini": GeminiMemoryProvider
    }
    
    def __init__(self):
        """Initialize sync manager."""
        self.memory_manager = SuperMemoryManager()
        self.conflict_resolver = ConflictResolver()
        self.providers: Dict[str, BaseMemoryProvider] = {}
        self._sync_history: Dict[str, SyncStatus] = {}
    
    async def register_provider(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a memory provider.
        
        Args:
            provider_name: Name of provider (claude, openai, gemini)
            api_key: API key for the provider
            config: Additional configuration
            
        Returns:
            True if registration successful
        """
        if provider_name not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = self.PROVIDER_CLASSES[provider_name]
        provider = provider_class(api_key=api_key, config=config)
        
        connected = await provider.connect()
        if connected:
            self.providers[provider_name] = provider
            return True
        
        return False
    
    async def unregister_provider(self, provider_name: str) -> None:
        """
        Unregister a provider.
        
        Args:
            provider_name: Name of provider to unregister
        """
        if provider_name in self.providers:
            await self.providers[provider_name].disconnect()
            del self.providers[provider_name]
    
    async def sync_all_providers(
        self,
        user_id: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        since: Optional[datetime] = None
    ) -> SyncStatus:
        """
        Synchronize with all registered providers.
        
        Args:
            user_id: User identifier
            direction: Sync direction
            since: Only sync memories after this time
            
        Returns:
            SyncStatus with results from all providers
        """
        import uuid
        sync_id = f"sync_{uuid.uuid4().hex[:12]}"
        
        status = SyncStatus(
            sync_id=sync_id,
            user_id=user_id,
            status="running",
            started_at=datetime.now()
        )
        
        self._sync_history[sync_id] = status
        
        # Get local memories
        await self.memory_manager.initialize()
        local_memories = await self.memory_manager.get_user_memories(user_id)
        
        # Sync with each provider in parallel
        tasks = []
        for provider_name, provider in self.providers.items():
            task = self._sync_with_provider(
                provider=provider,
                user_id=user_id,
                local_memories=local_memories,
                direction=direction,
                since=since
            )
            tasks.append((provider_name, task))
        
        # Wait for all syncs to complete
        for provider_name, task in tasks:
            try:
                result = await task
                status.provider_results[provider_name] = result
                status.total_memories_synced += (
                    result.memories_pulled + result.memories_pushed
                )
                if not result.success:
                    status.errors.extend(result.errors)
            except Exception as e:
                status.errors.append(f"{provider_name}: {str(e)}")
        
        # Set final status
        status.status = "completed" if not status.errors else "completed_with_errors"
        status.completed_at = datetime.now()
        
        return status
    
    async def sync_single_provider(
        self,
        provider_name: str,
        user_id: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        since: Optional[datetime] = None
    ) -> ProviderSyncResult:
        """
        Synchronize with a single provider.
        
        Args:
            provider_name: Provider to sync with
            user_id: User identifier
            direction: Sync direction
            since: Only sync memories after this time
            
        Returns:
            ProviderSyncResult
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider not registered: {provider_name}")
        
        provider = self.providers[provider_name]
        
        # Get local memories
        await self.memory_manager.initialize()
        local_memories = await self.memory_manager.get_user_memories(user_id)
        
        return await self._sync_with_provider(
            provider=provider,
            user_id=user_id,
            local_memories=local_memories,
            direction=direction,
            since=since
        )
    
    async def _sync_with_provider(
        self,
        provider: BaseMemoryProvider,
        user_id: str,
        local_memories: List[Dict[str, Any]],
        direction: SyncDirection,
        since: Optional[datetime]
    ) -> ProviderSyncResult:
        """
        Internal method to sync with a single provider.
        """
        result = ProviderSyncResult(
            provider_name=provider.provider_name,
            success=True,
            sync_started_at=datetime.now()
        )
        
        try:
            # Pull from provider
            if direction in [SyncDirection.PULL, SyncDirection.BIDIRECTIONAL]:
                pulled_memories = await provider.pull_memories(
                    user_id=user_id,
                    since=since
                )
                
                # Process pulled memories with conflict resolution
                for provider_memory in pulled_memories:
                    local_memory = provider.transform_from_provider_format(provider_memory)
                    
                    # Check for duplicates
                    is_duplicate = await self._check_duplicate(
                        user_id=user_id,
                        content=local_memory["content"]
                    )
                    
                    if is_duplicate:
                        result.memories_skipped += 1
                        continue
                    
                    # Check for conflicts
                    existing = await self._find_existing_memory(
                        user_id=user_id,
                        source_id=local_memory.get("source_id"),
                        source_provider=provider.provider_name
                    )
                    
                    if existing:
                        # Resolve conflict
                        resolved = self.conflict_resolver.resolve(
                            local=existing,
                            remote=local_memory
                        )
                        if resolved != existing:
                            # Update with resolved version
                            result.memories_updated += 1
                    else:
                        # Store new memory
                        from src.super_memory.memory_manager import Memory, MemoryType
                        memory = Memory(
                            content=local_memory["content"],
                            memory_type=MemoryType(local_memory.get("memory_type", "fact")),
                            category=local_memory.get("category"),
                            source_provider=provider.provider_name,
                            source_id=local_memory.get("source_id")
                        )
                        await self.memory_manager.store_memory(user_id, memory)
                        result.memories_pulled += 1
            
            # Push to provider
            if direction in [SyncDirection.PUSH, SyncDirection.BIDIRECTIONAL]:
                # Filter memories that originated from this provider
                memories_to_push = [
                    m for m in local_memories
                    if m.get("source_provider") != provider.provider_name
                ]
                
                pushed = await provider.push_memories(
                    user_id=user_id,
                    memories=memories_to_push
                )
                result.memories_pushed = pushed
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
        
        result.sync_completed_at = datetime.now()
        return result
    
    async def _check_duplicate(
        self,
        user_id: str,
        content: str
    ) -> bool:
        """Check if memory is a duplicate using similarity."""
        # Use embedding similarity to find duplicates
        similar = await self.memory_manager.search_memories(
            user_id=user_id,
            query=content,
            limit=1,
            min_confidence=settings.memory_similarity_threshold
        )
        
        if similar and similar[0].get("relevance_score", 0) > settings.memory_similarity_threshold:
            return True
        return False
    
    async def _find_existing_memory(
        self,
        user_id: str,
        source_id: Optional[str],
        source_provider: str
    ) -> Optional[Dict[str, Any]]:
        """Find existing memory by source ID."""
        if not source_id:
            return None
        
        # Query by source_id and source_provider
        # This would need a database query implementation
        return None
    
    def get_sync_status(self, sync_id: str) -> Optional[SyncStatus]:
        """Get status of a sync operation."""
        return self._sync_history.get(sync_id)
    
    def get_provider_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered providers."""
        return {
            name: provider.get_status()
            for name, provider in self.providers.items()
        }
    
    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self.providers.values():
            await provider.disconnect()
        self.providers.clear()
