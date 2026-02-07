"""
Agentic API Routes
Provides endpoints for agentic AI capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.api.auth import get_current_user
from src.agentic.orchestrator import AgentOrchestrator, OrchestratorResponse
from src.agentic.agents.product_selection_agent import ProductSelectionAgent
from src.agentic.agents.enquiry_management_agent import EnquiryManagementAgent
from src.agentic.multi_agent.coordinator import MultiAgentCoordinator, AgentRole
from src.agentic.hitl.approval_manager import ApprovalManager, ApprovalStatus, ActionType
from src.agentic.hitl.guardrails import Guardrails
from src.agentic.observability.tracer import AgentTracer
from src.agentic.observability.monitor import AgentMonitor
from src.agentic.retrieval.semantic_cache import SemanticCache


router = APIRouter(prefix="/agentic", tags=["Agentic AI"])

# Initialize components (in production, use dependency injection)
orchestrator = AgentOrchestrator()
product_selector = ProductSelectionAgent()
enquiry_manager = EnquiryManagementAgent()
coordinator = MultiAgentCoordinator()
approval_manager = ApprovalManager()
guardrails = Guardrails()
tracer = AgentTracer()
monitor = AgentMonitor()
cache = SemanticCache()


# Request/Response Models
class AgenticQueryRequest(BaseModel):
    """Request for agentic query processing."""
    query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class AgenticQueryResponse(BaseModel):
    """Response from agentic query processing."""
    response_text: str
    requires_user_input: bool = False
    clarifying_question: Optional[str] = None
    suggested_options: List[str] = []
    sources: List[Dict[str, Any]] = []
    actions_taken: List[str] = []
    validation_warnings: List[str] = []
    state: str = "idle"
    metadata: Dict[str, Any] = {}


class ProductSelectionRequest(BaseModel):
    """Request for guided product selection."""
    session_id: str
    user_input: Optional[str] = None
    action: str = "continue"  # start, continue, reset


class ProductSelectionResponse(BaseModel):
    """Response from product selection."""
    message: str
    question: Optional[str] = None
    options: List[str] = []
    recommendations: List[Dict[str, Any]] = []
    is_complete: bool = False
    requires_input: bool = True
    progress_percent: int = 0
    validation_warnings: List[str] = []


class EnquiryRequest(BaseModel):
    """Request for enquiry processing."""
    content: str
    from_email: str
    subject: str = ""
    customer_name: Optional[str] = None
    company: Optional[str] = None


class EnquiryResponse(BaseModel):
    """Response from enquiry processing."""
    enquiry_id: str
    classification: Dict[str, Any]
    instant_response: Optional[str] = None
    routed: bool
    routed_to: Optional[str] = None
    reference_number: str
    actions_taken: List[str]


class ApprovalRequestModel(BaseModel):
    """Request for approval submission."""
    action_type: str
    action_description: str
    action_payload: Dict[str, Any]
    priority: str = "normal"


class ApprovalDecisionModel(BaseModel):
    """Decision on an approval request."""
    decision: str  # approve, reject
    notes: str = ""


# Endpoints
@router.post("/query", response_model=AgenticQueryResponse)
async def process_agentic_query(
    request: AgenticQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Process a query through the agentic system.
    
    The orchestrator will:
    1. Analyze the query
    2. Identify missing parameters
    3. Execute tools as needed
    4. Validate results
    5. Generate response
    """
    # Check cache first
    cached = await cache.get(request.query)
    if cached:
        return AgenticQueryResponse(
            response_text=cached["response"],
            sources=cached.get("sources", []),
            metadata={"cache_hit": True}
        )
    
    # Check input guardrails
    input_results = await guardrails.check_input(request.query)
    blocked = [r for r in input_results if not r.passed and r.severity.value == "block"]
    if blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {blocked[0].message}"
        )
    
    # Process through orchestrator
    try:
        result = await orchestrator.process(
            query=request.query,
            session_id=request.session_id,
            user_id=current_user.get("user_id"),
            user_context={
                "role": current_user.get("role"),
                "department": current_user.get("department"),
                **(request.context or {})
            }
        )
        
        # Check output guardrails
        modified_response, output_results = await guardrails.check_output(result.response_text)
        
        # Record metrics
        background_tasks.add_task(
            monitor.record_request,
            latency_ms=result.metadata.get("processing_time_ms", 0),
            tokens_used=result.metadata.get("tokens_used", 0),
            tool_calls=len(result.actions_taken),
            tool_failures=0,
            retrievals=len(result.sources),
            cache_hit=False,
            error=False
        )
        
        # Cache successful response
        if result.state == "idle" and not result.requires_user_input:
            background_tasks.add_task(
                cache.set,
                request.query,
                modified_response,
                result.sources
            )
        
        return AgenticQueryResponse(
            response_text=modified_response,
            requires_user_input=result.requires_user_input,
            clarifying_question=result.clarifying_question,
            suggested_options=result.suggested_options,
            sources=result.sources,
            actions_taken=result.actions_taken,
            validation_warnings=result.validation_warnings,
            state=result.state.value,
            metadata=result.metadata
        )
        
    except Exception as e:
        monitor.record_request(
            latency_ms=0,
            tokens_used=0,
            tool_calls=0,
            tool_failures=1,
            retrievals=0,
            cache_hit=False,
            error=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/product-selection", response_model=ProductSelectionResponse)
async def guided_product_selection(
    request: ProductSelectionRequest
):
    """
    Guided product selection with targeted questions.
    
    This is NOT an open chatbot - it follows a structured
    decision tree to recommend products.
    
    Note: This endpoint is public for external customer use.
    """

    if request.action == "start":
        # Start new selection
        result = product_selector.start_selection(request.session_id)
    elif request.action == "reset":
        # Reset and start over
        product_selector.clear_session(request.session_id)
        result = product_selector.start_selection(request.session_id)
    else:
        # Continue with user input
        if not request.user_input:
            raise HTTPException(
                status_code=400,
                detail="user_input required for continue action"
            )
        result = product_selector.process_input(
            request.session_id,
            request.user_input
        )
    
    return ProductSelectionResponse(
        message=result.message,
        question=result.question,
        options=result.options,
        recommendations=result.recommendations,
        is_complete=result.is_complete,
        requires_input=result.requires_input,
        progress_percent=result.state._calculate_progress() if result.state else 0,
        validation_warnings=result.validation_warnings
    )


@router.get("/product-selection/{session_id}/state")
async def get_selection_state(
    session_id: str
):
    """Get current state of product selection session (public endpoint)."""

    state = product_selector.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return state.to_dict()


@router.post("/enquiry", response_model=EnquiryResponse)
async def process_enquiry(
    request: EnquiryRequest,
    background_tasks: BackgroundTasks
):
    """
    Process incoming customer enquiry.
    
    Classifies, routes, and optionally provides instant response.
    This endpoint can be called from external systems.
    """
    result = await enquiry_manager.process_enquiry(
        enquiry_content=request.content,
        from_email=request.from_email,
        subject=request.subject,
        customer_name=request.customer_name,
        company=request.company
    )
    
    return EnquiryResponse(
        enquiry_id=result.enquiry_id,
        classification=result.classification.to_dict(),
        instant_response=result.instant_response,
        routed=result.routed,
        routed_to=result.routed_to,
        reference_number=result.reference_number,
        actions_taken=result.actions_taken
    )


@router.post("/workflow")
async def execute_workflow(
    request: AgenticQueryRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Execute a multi-agent workflow for complex queries.
    
    Uses multiple specialized agents working together:
    - Planner: Decomposes the query
    - Researcher: Gathers information
    - Writer: Synthesizes response
    - Reviewer: Validates output
    """
    result = await coordinator.execute_workflow(
        query=request.query,
        context=request.context,
        user_id=current_user.get("user_id")
    )
    
    return result.to_dict()


# Approval Endpoints
@router.post("/approval/request")
async def request_approval(
    request: ApprovalRequestModel,
    current_user: Dict = Depends(get_current_user)
):
    """Request approval for a sensitive action."""
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action type: {request.action_type}"
        )
    
    approval_request = await approval_manager.request_approval(
        action_type=action_type,
        action_description=request.action_description,
        action_payload=request.action_payload,
        requestor_id=current_user.get("user_id"),
        requestor_context={"role": current_user.get("role")},
        priority=request.priority
    )
    
    return approval_request.to_dict()


@router.get("/approval/pending")
async def get_pending_approvals(
    current_user: Dict = Depends(get_current_user)
):
    """Get all pending approval requests."""
    # Check if user has approval authority
    if current_user.get("role") not in ["manager", "admin", "supervisor"]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view approvals"
        )
    
    pending = approval_manager.get_pending_requests()
    return [r.to_dict() for r in pending]


@router.post("/approval/{request_id}/decide")
async def decide_approval(
    request_id: str,
    decision: ApprovalDecisionModel,
    current_user: Dict = Depends(get_current_user)
):
    """Approve or reject an approval request."""
    # Check permissions
    if current_user.get("role") not in ["manager", "admin", "supervisor"]:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to decide approvals"
        )
    
    if decision.decision == "approve":
        success, result = await approval_manager.approve(
            request_id=request_id,
            reviewer_id=current_user.get("user_id"),
            notes=decision.notes
        )
    elif decision.decision == "reject":
        success = await approval_manager.reject(
            request_id=request_id,
            reviewer_id=current_user.get("user_id"),
            reason=decision.notes
        )
        result = None
    else:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approve' or 'reject'"
        )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to process approval decision"
        )
    
    return {"status": decision.decision, "result": result}


# Observability Endpoints
@router.get("/traces")
async def get_traces(
    limit: int = 50,
    current_user: Dict = Depends(get_current_user)
):
    """Get recent traces for debugging."""
    if current_user.get("role") not in ["admin", "engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    traces = tracer.get_recent_traces(limit)
    return [t.to_dict() for t in traces]


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get a specific trace."""
    if current_user.get("role") not in ["admin", "engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    trace = tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return trace.to_dict()


@router.get("/metrics")
async def get_metrics(
    hours: int = 1,
    current_user: Dict = Depends(get_current_user)
):
    """Get agent metrics summary."""
    if current_user.get("role") not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return monitor.get_summary(hours)


@router.get("/alerts")
async def get_alerts(
    unacknowledged_only: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """Get monitoring alerts."""
    if current_user.get("role") not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    alerts = monitor.get_alerts(unacknowledged_only=unacknowledged_only)
    return [a.to_dict() for a in alerts]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Acknowledge an alert."""
    if current_user.get("role") not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    success = monitor.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"status": "acknowledged"}


@router.get("/cache/stats")
async def get_cache_stats(
    current_user: Dict = Depends(get_current_user)
):
    """Get semantic cache statistics."""
    if current_user.get("role") not in ["admin", "engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return cache.get_stats().to_dict()


@router.post("/cache/invalidate")
async def invalidate_cache(
    pattern: str,
    current_user: Dict = Depends(get_current_user)
):
    """Invalidate cache entries matching a pattern."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    
    count = cache.invalidate_pattern(pattern)
    return {"invalidated": count}


@router.get("/session/{session_id}")
async def get_session_state(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get session state from orchestrator."""
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return state


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Clear a session."""
    success = orchestrator.clear_session(session_id)
    return {"cleared": success}
