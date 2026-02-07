"""
LangGraph State Definitions for the Agent Orchestrator.
Defines the shared state structure used across all graph nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class AgentGraphState(TypedDict):
    """
    Shared state across all nodes in the agent graph.
    
    This state is passed between nodes and accumulates results
    from each processing step. Use Annotated with 'add' for 
    fields that should accumulate (like trace_steps, tool_results).
    """
    # === Input Fields ===
    query: str
    session_id: str
    user_id: Optional[str]
    user_context: Optional[Dict[str, Any]]
    
    # === Processing State ===
    current_state: str  # STARTED, ANALYZING, EXECUTING, VALIDATING, COMPLETE, FAILED
    
    # === Analysis Results ===
    intent: Optional[str]
    complexity: Optional[str]
    extracted_parameters: Dict[str, Any]
    missing_parameters: List[str]
    suggested_values: Dict[str, List[str]]
    
    # === Execution Tracking ===
    # Use Annotated[..., add] to accumulate results across iterations
    tool_results: Annotated[List[Dict[str, Any]], add]
    validation_results: List[Dict[str, Any]]
    specialist_used: Optional[str]
    tier_used: Optional[str]
    
    # === Output Fields ===
    response_text: str
    requires_user_input: bool
    clarifying_question: Optional[str]
    suggested_options: List[str]
    sources: List[Dict[str, Any]]
    
    # === Debugging & Tracing ===
    trace_steps: Annotated[List[Dict[str, Any]], add]
    errors: List[str]
    warnings: List[str]
    
    # === Metadata ===
    started_at: str
    iteration_count: int


def create_initial_state(
    query: str,
    session_id: str,
    user_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
) -> AgentGraphState:
    """
    Create initial state for a new graph execution.
    
    Args:
        query: User's input query
        session_id: Session identifier
        user_id: Optional user identifier
        user_context: Optional additional context
        
    Returns:
        Initialized AgentGraphState
    """
    from datetime import datetime
    
    return AgentGraphState(
        # Input
        query=query,
        session_id=session_id,
        user_id=user_id,
        user_context=user_context or {},
        
        # Processing
        current_state="STARTED",
        
        # Analysis
        intent=None,
        complexity=None,
        extracted_parameters={},
        missing_parameters=[],
        suggested_values={},
        
        # Execution
        tool_results=[],
        validation_results=[],
        specialist_used=None,
        tier_used=None,
        
        # Output
        response_text="",
        requires_user_input=False,
        clarifying_question=None,
        suggested_options=[],
        sources=[],
        
        # Debugging
        trace_steps=[],
        errors=[],
        warnings=[],
        
        # Metadata
        started_at=datetime.now().isoformat(),
        iteration_count=0,
    )
