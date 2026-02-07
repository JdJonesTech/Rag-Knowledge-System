"""
Base Tool Interface
Abstract base class for all agentic tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    status: ToolStatus
    data: Any = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "data": self.data,
            "sources": self.sources,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL]


class BaseTool(ABC):
    """
    Abstract base class for tools.
    
    All tools must implement:
    - name: Unique identifier
    - description: What the tool does (for the orchestrator)
    - execute: Main execution method
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize tool.
        
        Args:
            name: Tool identifier
            description: Tool description
        """
        self.name = name
        self.description = description
        self._call_count = 0
        self._last_called: Optional[datetime] = None
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            query: The query/request
            parameters: Parameters for the tool
            intent: Optional intent context
            
        Returns:
            ToolResult with data or error
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's parameter schema.
        
        Returns:
            JSON schema for parameters
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.get_schema(),
            "call_count": self._call_count,
            "last_called": self._last_called.isoformat() if self._last_called else None
        }
    
    async def __call__(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Callable interface for the tool.
        Tracks execution stats.
        """
        import time
        start = time.time()
        
        self._call_count += 1
        self._last_called = datetime.now()
        
        try:
            result = await self.execute(query, parameters, intent)
            result.execution_time_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )
