"""
Conflict Resolver
Handles conflicts when syncing memories between providers.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class ConflictStrategy(str, Enum):
    """Conflict resolution strategies."""
    REMOTE_WINS = "remote_wins"  # Provider version wins
    LOCAL_WINS = "local_wins"  # Local version wins
    NEWER_WINS = "newer_wins"  # Most recently updated wins
    MERGE = "merge"  # Attempt to merge both versions
    HIGHEST_CONFIDENCE = "highest_confidence"  # Highest confidence score wins
    MANUAL = "manual"  # Flag for manual resolution


@dataclass
class ConflictRecord:
    """Record of a conflict."""
    conflict_id: str
    local_memory: Dict[str, Any]
    remote_memory: Dict[str, Any]
    strategy_used: ConflictStrategy
    resolved_memory: Dict[str, Any]
    resolved_at: datetime
    auto_resolved: bool


class ConflictResolver:
    """
    Resolves conflicts between local and remote memories.
    Supports multiple resolution strategies.
    """
    
    def __init__(
        self,
        default_strategy: ConflictStrategy = ConflictStrategy.NEWER_WINS
    ):
        """
        Initialize conflict resolver.
        
        Args:
            default_strategy: Default resolution strategy
        """
        self.default_strategy = default_strategy
        self.conflict_history: List[ConflictRecord] = []
        self.pending_manual_conflicts: List[Dict[str, Any]] = []
    
    def resolve(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any],
        strategy: Optional[ConflictStrategy] = None
    ) -> Dict[str, Any]:
        """
        Resolve a conflict between local and remote versions.
        
        Args:
            local: Local memory version
            remote: Remote memory version
            strategy: Resolution strategy to use
            
        Returns:
            Resolved memory
        """
        strategy = strategy or self.default_strategy
        
        # Check if content is identical (no actual conflict)
        if self._content_matches(local, remote):
            return self._merge_metadata(local, remote)
        
        # Apply resolution strategy
        if strategy == ConflictStrategy.REMOTE_WINS:
            resolved = self._resolve_remote_wins(local, remote)
        elif strategy == ConflictStrategy.LOCAL_WINS:
            resolved = self._resolve_local_wins(local, remote)
        elif strategy == ConflictStrategy.NEWER_WINS:
            resolved = self._resolve_newer_wins(local, remote)
        elif strategy == ConflictStrategy.MERGE:
            resolved = self._resolve_merge(local, remote)
        elif strategy == ConflictStrategy.HIGHEST_CONFIDENCE:
            resolved = self._resolve_highest_confidence(local, remote)
        elif strategy == ConflictStrategy.MANUAL:
            self._queue_manual_resolution(local, remote)
            resolved = local  # Keep local until manually resolved
        else:
            resolved = local
        
        # Record conflict
        self._record_conflict(
            local=local,
            remote=remote,
            strategy=strategy,
            resolved=resolved
        )
        
        return resolved
    
    def _content_matches(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> bool:
        """Check if content is essentially the same."""
        local_content = local.get("content", "").strip().lower()
        remote_content = remote.get("content", "").strip().lower()
        
        # Exact match
        if local_content == remote_content:
            return True
        
        # Check for high similarity (simple approach)
        if len(local_content) > 0 and len(remote_content) > 0:
            # Calculate Jaccard similarity on words
            local_words = set(local_content.split())
            remote_words = set(remote_content.split())
            
            if not local_words or not remote_words:
                return False
            
            intersection = len(local_words & remote_words)
            union = len(local_words | remote_words)
            
            if union > 0 and intersection / union > 0.9:
                return True
        
        return False
    
    def _merge_metadata(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge metadata from both versions when content matches."""
        merged = local.copy()
        
        # Take higher scores
        merged["importance_score"] = max(
            local.get("importance_score", 0),
            remote.get("importance_score", 0)
        )
        merged["confidence_score"] = max(
            local.get("confidence_score", 0),
            remote.get("confidence_score", 0)
        )
        
        # Merge tags
        local_tags = set(local.get("tags", []))
        remote_tags = set(remote.get("tags", []))
        merged["tags"] = list(local_tags | remote_tags)
        
        # Track sync
        merged["last_synced_at"] = datetime.now().isoformat()
        
        return merged
    
    def _resolve_remote_wins(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remote version wins."""
        resolved = remote.copy()
        
        # Preserve local ID if exists
        if local.get("memory_id"):
            resolved["memory_id"] = local["memory_id"]
        
        # Mark as synced
        resolved["last_synced_at"] = datetime.now().isoformat()
        resolved["sync_source"] = "remote"
        
        return resolved
    
    def _resolve_local_wins(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Local version wins."""
        resolved = local.copy()
        resolved["last_synced_at"] = datetime.now().isoformat()
        resolved["sync_source"] = "local"
        return resolved
    
    def _resolve_newer_wins(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Most recently updated version wins."""
        local_updated = self._get_timestamp(local)
        remote_updated = self._get_timestamp(remote)
        
        if remote_updated and local_updated:
            if remote_updated > local_updated:
                return self._resolve_remote_wins(local, remote)
            else:
                return self._resolve_local_wins(local, remote)
        elif remote_updated:
            return self._resolve_remote_wins(local, remote)
        else:
            return self._resolve_local_wins(local, remote)
    
    def _resolve_merge(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to merge both versions."""
        # Start with local as base
        merged = local.copy()
        
        # Merge content if different
        local_content = local.get("content", "")
        remote_content = remote.get("content", "")
        
        if local_content != remote_content:
            # Simple merge: keep both with separator
            # In production, use more sophisticated merging
            merged["content"] = f"{local_content}\n---\n[Additional info]: {remote_content}"
        
        # Take higher scores
        merged["importance_score"] = max(
            local.get("importance_score", 0),
            remote.get("importance_score", 0)
        )
        merged["confidence_score"] = max(
            local.get("confidence_score", 0),
            remote.get("confidence_score", 0)
        )
        
        # Merge tags
        local_tags = set(local.get("tags", []))
        remote_tags = set(remote.get("tags", []))
        merged["tags"] = list(local_tags | remote_tags)
        
        # Merge metadata
        local_meta = local.get("metadata", {})
        remote_meta = remote.get("metadata", {})
        merged["metadata"] = {**local_meta, **remote_meta}
        
        merged["last_synced_at"] = datetime.now().isoformat()
        merged["sync_source"] = "merged"
        
        return merged
    
    def _resolve_highest_confidence(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Version with highest confidence wins."""
        local_confidence = local.get("confidence_score", 0)
        remote_confidence = remote.get("confidence_score", 0)
        
        if remote_confidence > local_confidence:
            return self._resolve_remote_wins(local, remote)
        else:
            return self._resolve_local_wins(local, remote)
    
    def _queue_manual_resolution(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any]
    ) -> None:
        """Queue conflict for manual resolution."""
        import uuid
        self.pending_manual_conflicts.append({
            "conflict_id": f"conflict_{uuid.uuid4().hex[:8]}",
            "local": local,
            "remote": remote,
            "created_at": datetime.now().isoformat()
        })
    
    def _get_timestamp(self, memory: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from memory."""
        for field in ["updated_at", "created_at", "last_synced_at"]:
            if memory.get(field):
                ts = memory[field]
                if isinstance(ts, datetime):
                    return ts
                elif isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue
        return None
    
    def _record_conflict(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any],
        strategy: ConflictStrategy,
        resolved: Dict[str, Any]
    ) -> None:
        """Record conflict for auditing."""
        import uuid
        record = ConflictRecord(
            conflict_id=f"conflict_{uuid.uuid4().hex[:8]}",
            local_memory=local,
            remote_memory=remote,
            strategy_used=strategy,
            resolved_memory=resolved,
            resolved_at=datetime.now(),
            auto_resolved=strategy != ConflictStrategy.MANUAL
        )
        self.conflict_history.append(record)
        
        # Keep only last 1000 records
        if len(self.conflict_history) > 1000:
            self.conflict_history = self.conflict_history[-1000:]
    
    def get_pending_conflicts(self) -> List[Dict[str, Any]]:
        """Get conflicts pending manual resolution."""
        return self.pending_manual_conflicts
    
    def resolve_manual_conflict(
        self,
        conflict_id: str,
        resolution: Dict[str, Any]
    ) -> bool:
        """
        Manually resolve a pending conflict.
        
        Args:
            conflict_id: Conflict identifier
            resolution: Resolved memory
            
        Returns:
            True if conflict was found and resolved
        """
        for i, conflict in enumerate(self.pending_manual_conflicts):
            if conflict["conflict_id"] == conflict_id:
                # Record resolution
                self._record_conflict(
                    local=conflict["local"],
                    remote=conflict["remote"],
                    strategy=ConflictStrategy.MANUAL,
                    resolved=resolution
                )
                
                # Remove from pending
                self.pending_manual_conflicts.pop(i)
                return True
        
        return False
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        total = len(self.conflict_history)
        
        strategy_counts = {}
        for record in self.conflict_history:
            strategy = record.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_conflicts": total,
            "pending_manual": len(self.pending_manual_conflicts),
            "by_strategy": strategy_counts,
            "auto_resolved_rate": (
                sum(1 for r in self.conflict_history if r.auto_resolved) / total
                if total > 0 else 0
            )
        }
