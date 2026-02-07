"""
Human-in-the-Loop (HITL) Module
Implements approval workflows for sensitive actions.
"""

from src.agentic.hitl.approval_manager import ApprovalManager, ApprovalRequest, ApprovalStatus
from src.agentic.hitl.guardrails import Guardrails, GuardrailResult, AggregatedGuardrailResult

__all__ = [
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalStatus",
    "Guardrails",
    "GuardrailResult",
    "AggregatedGuardrailResult"
]
