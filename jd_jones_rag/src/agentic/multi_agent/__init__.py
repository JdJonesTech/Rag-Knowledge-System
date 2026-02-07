"""
Multi-Agent Coordination Module
Manages multiple specialized agents working together.
"""

from src.agentic.multi_agent.coordinator import MultiAgentCoordinator
from src.agentic.multi_agent.agent_registry import AgentRegistry

__all__ = [
    "MultiAgentCoordinator",
    "AgentRegistry"
]
