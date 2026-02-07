"""
Base Tool Exports - Backward compatibility alias.
Re-exports from base_tool.py
"""

from src.agentic.tools.base_tool import (
    ToolStatus,
    ToolResult,
    BaseTool
)

# Create ToolInput as an alias for compatibility with tests
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ToolInput:
    """Input for a tool execution."""
    tool_name: str
    parameters: Dict[str, Any]

__all__ = [
    "ToolStatus",
    "ToolResult",
    "BaseTool",
    "ToolInput"
]
