"""
Super Memory Models - Backward compatibility module.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


class MemoryType(str, Enum):
    """Types of memories the system can store."""
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    INTERACTION = "interaction"
    SUMMARY = "summary"


class MemoryPriority(str, Enum):
    """Priority levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


__all__ = [
    "MemoryType",
    "MemoryPriority",
    "Memory"
]
