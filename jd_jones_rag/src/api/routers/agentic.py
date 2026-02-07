"""
Agentic AI API Router
Provides endpoints for the agentic AI system.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from src.auth.jwt_handler import get_current_user
from src.auth.rbac import UserRole, check_permission


# Initialize router
router = APIRouter(prefix="/agentic", tags=["Agentic AI"])


# ============= Request/Response Models =============

class AgenticQueryRequest(BaseModel):
    """Request for agentic query processing."""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    use_multi_agent: bool = Field(False, description="Use multi-agent coordination")
    workflow: Optional[str] = Field(None, description="Workflow template name")


class AgenticQueryResponse(BaseModel):
    """Response from agentic query processing."""
    success: bool
    response_text: str
    session_id: str
    sources: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    requires_input: bool = False
    clarifying_question: Optional[str] = None
    suggested_options: List[str] = []
    validation_warnings: List[str] = []
    metadata: Dict[str, Any] = {}


class ProductSelectionRequest(BaseModel):
    """Request for product selection."""
    session_id: Optional[str] = None
    user_input: Optional[str] = None
    start_new: bool = False
    # Structured answers from wizard
    application_type: Optional[str] = None
    industry: Optional[str] = None
    temperature_range: Optional[str] = None
    pressure_range: Optional[str] = None
    media_type: Optional[str] = None
    required_certifications: Optional[List[str]] = None


class ProductSelectionResponse(BaseModel):
    """Response from product selection."""
    session_id: str
    stage: str
    message: str
    question: Optional[str] = None
    options: List[str] = []
    recommendations: List[Dict[str, Any]] = []
    is_complete: bool = False
    requires_input: bool = True
    progress_percent: int = 0


# Models for the agentic enquiry processing endpoint
class ProcessEnquiryRequest(BaseModel):
    """Request for agentic enquiry processing."""
    content: str
    from_email: str
    subject: Optional[str] = ""
    customer_name: Optional[str] = None
    company: Optional[str] = None


class ProcessEnquiryResponse(BaseModel):
    """Response from agentic enquiry processing."""
    enquiry_id: str
    classification: Dict[str, Any]
    instant_response: Optional[str] = None
    routed: bool = False
    routed_to: Optional[str] = None
    reference_number: str
    actions_taken: List[str] = []





class ApprovalRequest(BaseModel):
    """Request for action approval."""
    request_id: str
    action: str  # approve, reject, cancel
    notes: Optional[str] = ""
    reason: Optional[str] = ""


class ApprovalResponse(BaseModel):
    """Response from approval action."""
    request_id: str
    status: str
    message: str


# ============= Orchestrator Instance =============

# Global instances (initialized lazily)
_orchestrator = None
_product_selection_agent = None
_enquiry_agent = None
_approval_manager = None
_tracer = None
_monitor = None


def get_orchestrator():
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from src.agentic.orchestrator import AgentOrchestrator
        from src.agentic.tools.vector_search_tool import VectorSearchTool, ProductDatabaseTool
        from src.agentic.tools.sql_query_tool import SQLQueryTool
        from src.agentic.tools.compliance_checker_tool import ComplianceCheckerTool
        from src.agentic.tools.email_tool import EmailTool
        from src.agentic.tools.crm_tool import CRMTool
        from src.agentic.tools.document_generator_tool import DocumentGeneratorTool
        from src.agentic.tools.jira_tool import JiraTool
        from src.agentic.tools.sharepoint_tool import SharePointTool
        from src.agentic.tools.web_search_tool import WebSearchTool
        from src.agentic.tools.code_interpreter_tool import CodeInterpreterTool
        from src.agentic.tools.slack_tool import SlackTool
        
        _orchestrator = AgentOrchestrator()
        
        # Register all tools
        tools = [
            # Core RAG tools
            VectorSearchTool(),
            ProductDatabaseTool(),
            SQLQueryTool(),
            
            # Communication tools
            EmailTool(),
            SlackTool(),
            
            # Enterprise integrations
            CRMTool(),
            JiraTool(),
            SharePointTool(),
            
            # Utility tools
            WebSearchTool(),
            CodeInterpreterTool(),
            DocumentGeneratorTool(),
            ComplianceCheckerTool()
        ]
        
        for tool in tools:
            _orchestrator.register_tool(tool.name, tool, tool.description)
    
    return _orchestrator


def get_product_selection_agent():
    """Get or create product selection agent."""
    global _product_selection_agent
    if _product_selection_agent is None:
        from src.agentic.agents.product_selection_agent import ProductSelectionAgent
        _product_selection_agent = ProductSelectionAgent()
    return _product_selection_agent


def get_enquiry_agent():
    """Get or create enquiry management agent."""
    global _enquiry_agent
    if _enquiry_agent is None:
        from src.agentic.agents.enquiry_management_agent import EnquiryManagementAgent
        _enquiry_agent = EnquiryManagementAgent()
    return _enquiry_agent


def get_approval_manager():
    """Get or create approval manager."""
    global _approval_manager
    if _approval_manager is None:
        from src.agentic.hitl.approval_manager import ApprovalManager
        _approval_manager = ApprovalManager()
    return _approval_manager


def get_tracer():
    """Get or create tracer."""
    global _tracer
    if _tracer is None:
        from src.agentic.observability.tracer import AgentTracer
        _tracer = AgentTracer()
    return _tracer


def get_monitor():
    """Get or create monitor."""
    global _monitor
    if _monitor is None:
        from src.agentic.observability.monitor import AgentMonitor
        _monitor = AgentMonitor()
    return _monitor


# ============= Endpoints =============

@router.post("/query", response_model=AgenticQueryResponse)
async def process_agentic_query(
    request: AgenticQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Process a query through the agentic AI system.
    
    Features:
    - Multi-step reasoning
    - Tool orchestration
    - Self-correction
    - Source citations
    """
    import uuid
    import time
    
    orchestrator = get_orchestrator()
    monitor = get_monitor()
    tracer = get_tracer()
    
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    try:
        # Start tracing
        tracer.start_trace(
            name="agentic_query",
            user_id=current_user.get("user_id"),
            session_id=session_id
        )
        
        # Process through orchestrator
        result = await orchestrator.process(
            query=request.query,
            session_id=session_id,
            user_id=current_user.get("user_id"),
            user_context={
                "role": current_user.get("role"),
                "department": current_user.get("department"),
                **request.context
            }
        )
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        monitor.record_request(
            latency_ms=latency_ms,
            tokens_used=result.metadata.get("tokens", 0),
            tool_calls=len(result.actions_taken),
            tool_failures=0,
            retrievals=len(result.sources),
            cache_hit=False,
            error=False
        )
        
        return AgenticQueryResponse(
            success=True,
            response_text=result.response_text,
            session_id=session_id,
            sources=result.sources,
            tools_used=result.actions_taken,
            requires_input=result.requires_user_input,
            clarifying_question=result.clarifying_question,
            suggested_options=result.suggested_options,
            validation_warnings=result.validation_warnings,
            metadata=result.metadata
        )
        
    except Exception as e:
        monitor.record_request(
            latency_ms=(time.time() - start_time) * 1000,
            tokens_used=0,
            tool_calls=0,
            tool_failures=1,
            retrievals=0,
            cache_hit=False,
            error=True
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracer.end_trace()


@router.post("/product-selection", response_model=ProductSelectionResponse)
async def guided_product_selection(
    request: ProductSelectionRequest
):
    """
    Guided product selection with targeted questions.
    
    This is a GUIDED experience, not open-ended:
    - Asks one question at a time
    - Provides options where possible
    - Validates recommendations against standards
    
    Note: Public endpoint for external customer use.
    """

    import uuid
    
    agent = get_product_selection_agent()
    
    session_id = request.session_id or f"ps_{uuid.uuid4().hex[:12]}"
    
    try:
        # Check if structured answers are provided (from wizard)
        if request.application_type:
            # Direct recommendation mode - all answers provided at once
            recommendations = agent.get_recommendations_from_answers({
                'application_type': request.application_type,
                'industry': request.industry,
                'temperature_range': request.temperature_range,
                'pressure_range': request.pressure_range,
                'media_type': request.media_type,
                'required_certifications': request.required_certifications or []
            })
            
            return ProductSelectionResponse(
                session_id=session_id,
                stage="complete",
                message="Based on your requirements, here are our recommended products:",
                question=None,
                options=[],
                recommendations=recommendations,
                is_complete=True,
                requires_input=False,
                progress_percent=100
            )
        
        # Original interactive mode
        if request.start_new or request.session_id is None:
            # Start new selection
            response = agent.start_selection(session_id)
        else:
            # Process user input
            response = agent.process_input(session_id, request.user_input or "")
        
        return ProductSelectionResponse(
            session_id=response.state.session_id,
            stage=response.state.current_stage.value,
            message=response.message,
            question=response.question,
            options=response.options,
            recommendations=response.recommendations,
            is_complete=response.is_complete,
            requires_input=response.requires_input,
            progress_percent=response.state._calculate_progress()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product-selection/{session_id}")
async def get_selection_state(
    session_id: str
):
    """Get current state of a product selection session (public endpoint)."""

    agent = get_product_selection_agent()
    state = agent.get_session(session_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return state.to_dict()


@router.delete("/product-selection/{session_id}")
async def clear_selection_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Clear a product selection session."""
    agent = get_product_selection_agent()
    
    if agent.clear_session(session_id):
        return {"message": "Session cleared", "session_id": session_id}
    
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/process-enquiry")
async def process_enquiry(
    request: ProcessEnquiryRequest,
    background_tasks: BackgroundTasks
):
    """
    Process an incoming enquiry.
    
    Features:
    - Auto-classification
    - Instant response for FAQs
    - Smart routing to appropriate team
    - CRM logging
    - Acknowledgment email
    """
    agent = get_enquiry_agent()
    
    try:
        result = await agent.process_enquiry(
            enquiry_content=request.content,
            from_email=request.from_email,
            subject=request.subject or "",
            customer_name=request.customer_name,
            company=request.company
        )
        
        return ProcessEnquiryResponse(
            enquiry_id=result.enquiry_id,
            classification=result.classification.to_dict(),
            instant_response=result.instant_response,
            routed=result.routed,
            routed_to=result.routed_to,
            reference_number=result.reference_number,
            actions_taken=result.actions_taken
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/approvals/pending")
async def get_pending_approvals(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get pending approval requests."""
    # Check permission
    if current_user.get("role") not in ["manager", "admin", "finance_manager", "legal"]:
        raise HTTPException(status_code=403, detail="Not authorized to view approvals")
    
    manager = get_approval_manager()
    
    from src.agentic.hitl.approval_manager import ActionCategory
    
    cat = None
    if category:
        try:
            cat = ActionCategory(category)
        except ValueError:
            pass
    
    pending = manager.get_pending_requests(category=cat)
    
    return {
        "pending_count": len(pending),
        "requests": [r.to_dict() for r in pending]
    }


@router.post("/approvals/{request_id}")
async def process_approval(
    request_id: str,
    request: ApprovalRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process an approval request (approve/reject/cancel)."""
    manager = get_approval_manager()
    
    try:
        if request.action == "approve":
            result = await manager.approve(
                request_id=request_id,
                approved_by=current_user.get("user_id", "unknown"),
                approver_role=current_user.get("role", "employee"),
                notes=request.notes or ""
            )
            message = "Request approved"
        
        elif request.action == "reject":
            if not request.reason:
                raise HTTPException(status_code=400, detail="Rejection reason required")
            
            result = await manager.reject(
                request_id=request_id,
                rejected_by=current_user.get("user_id", "unknown"),
                reason=request.reason
            )
            message = "Request rejected"
        
        elif request.action == "cancel":
            result = manager.cancel(
                request_id=request_id,
                cancelled_by=current_user.get("user_id", "unknown")
            )
            message = "Request cancelled"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return ApprovalResponse(
            request_id=result.request_id,
            status=result.status.value,
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats")
async def get_agentic_stats(
    current_user: dict = Depends(get_current_user)
):
    """Get agentic system statistics."""
    monitor = get_monitor()
    tracer = get_tracer()
    approval_manager = get_approval_manager()
    
    return {
        "performance": monitor.get_summary(hours=24),
        "tracing": tracer.get_stats(),
        "approvals": approval_manager.get_stats()
    }


@router.get("/traces")
async def get_recent_traces(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get recent execution traces for debugging."""
    # Admin only
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    tracer = get_tracer()
    traces = tracer.get_recent_traces(limit=limit)
    
    return {
        "count": len(traces),
        "traces": [t.to_dict() for t in traces]
    }


@router.get("/alerts")
async def get_alerts(
    level: Optional[str] = None,
    unacknowledged_only: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Get monitoring alerts."""
    # Manager+ only
    if current_user.get("role") not in ["manager", "admin"]:
        raise HTTPException(status_code=403, detail="Manager access required")
    
    monitor = get_monitor()
    
    from src.agentic.observability.monitor import AlertLevel
    
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel(level)
        except ValueError:
            pass
    
    alerts = monitor.get_alerts(level=alert_level, unacknowledged_only=unacknowledged_only)
    
    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts]
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an alert."""
    if current_user.get("role") not in ["manager", "admin"]:
        raise HTTPException(status_code=403, detail="Manager access required")
    
    monitor = get_monitor()
    
    if monitor.acknowledge_alert(alert_id):
        return {"message": "Alert acknowledged", "alert_id": alert_id}
    
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/session/{session_id}")
async def get_session_state(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get current state of an orchestrator session."""
    orchestrator = get_orchestrator()
    state = orchestrator.get_session_state(session_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return state


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Clear an orchestrator session."""
    orchestrator = get_orchestrator()
    
    if orchestrator.clear_session(session_id):
        return {"message": "Session cleared", "session_id": session_id}
    
    raise HTTPException(status_code=404, detail="Session not found")


# ============= Thought Trace & Specialist Endpoints =============

# Global specialist registry
_specialist_registry = None

def get_specialist_registry():
    """Get or create specialist registry."""
    global _specialist_registry
    if _specialist_registry is None:
        from src.agentic.agents.specialized_agents import SpecialistAgentRegistry
        _specialist_registry = SpecialistAgentRegistry()
        
        # Register tools with specialists
        orchestrator = get_orchestrator()
        tools = {}
        for name, tool_info in orchestrator.tools.items():
            if isinstance(tool_info, dict) and "function" in tool_info:
                tools[name] = tool_info["function"]
        _specialist_registry.register_tools(tools)
    
    return _specialist_registry


class QueryWithTraceRequest(BaseModel):
    """Request for query with full thought trace."""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    include_full_trace: bool = Field(True, description="Include detailed reasoning trace")


class QueryWithTraceResponse(BaseModel):
    """Response with full thought trace."""
    success: bool
    response_text: str
    session_id: str
    thought_trace: Dict[str, Any] = {}
    specialist_used: Optional[str] = None
    delegation_reason: Optional[str] = None
    tools_used: List[str] = []
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


@router.post("/query-with-trace", response_model=QueryWithTraceResponse)
async def query_with_trace(
    request: QueryWithTraceRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Process a query with full thought trace visibility.
    
    This endpoint returns the complete reasoning process including:
    - Each thought step the agent takes
    - Actions/tools executed
    - Observations from tools
    - Delegation decisions to specialist agents
    - Final synthesis
    """
    import uuid
    import time
    from src.agentic.agents.specialized_agents import AgentThoughtTrace
    
    orchestrator = get_orchestrator()
    specialist_registry = get_specialist_registry()
    tracer = get_tracer()
    
    session_id = request.session_id or f"trace_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    # Create thought trace
    trace = AgentThoughtTrace(
        trace_id=f"thought_{uuid.uuid4().hex[:12]}",
        agent_name="Orchestrator",
        query=request.query
    )
    
    specialist_used = None
    delegation_reason = None
    
    try:
        # Step 1: Analyze query intent
        trace.add_thought("Analyzing query to determine intent and required parameters...")
        
        router_agent = orchestrator.router
        analysis = await router_agent.analyze(
            query=request.query,
            context=request.context
        )
        
        trace.add_observation(
            f"Intent: {analysis.intent.value}, Confidence: {analysis.confidence:.2f}",
            source="RouterAgent"
        )
        
        # Step 2: Check if specialist should handle this
        trace.add_thought("Evaluating if a specialized agent should handle this query...")
        
        specialist_domain = specialist_registry.determine_specialist(
            query=request.query,
            intent=analysis.intent.value if analysis.intent else ""
        )
        
        if specialist_domain:
            specialist = specialist_registry.get_specialist(specialist_domain)
            specialist_used = specialist_domain.value
            delegation_reason = f"Query matches {specialist_domain.value} domain expertise"
            
            trace.add_delegation(
                specialist_name=specialist_domain.value,
                reason=delegation_reason
            )
            
            # Execute with specialist
            trace.add_thought(f"Delegating to {specialist_domain.value} specialist for precise response...")
            
            result = await specialist.execute(
                query=request.query,
                context={
                    "intent": analysis.intent.value if analysis.intent else "general",
                    "parameters": analysis.extracted_parameters,
                    "user_context": {"role": current_user.get("role")}
                }
            )
            
            # Add specialist's trace steps
            for step in result.trace:
                if step.step_type.value == "thought":
                    trace.add_thought(step.content)
                elif step.step_type.value == "action":
                    trace.add_action(step.tool_name or "unknown", step.tool_input or {})
                elif step.step_type.value == "observation":
                    trace.add_observation(step.content, step.tool_name)
            
            trace.final_answer = result.final_answer
            trace.tools_used = result.tools_used
            
        else:
            # Use main orchestrator
            trace.add_decision("No specialist needed, using main orchestrator")
            
            result = await orchestrator.process(
                query=request.query,
                session_id=session_id,
                user_id=current_user.get("user_id"),
                user_context={
                    "role": current_user.get("role"),
                    **request.context
                }
            )
            
            trace.add_thought("Processing through main orchestrator pipeline...")
            trace.add_observation(f"Actions taken: {', '.join(result.actions_taken)}")
            trace.final_answer = result.response_text
            trace.tools_used = result.actions_taken
        
        trace.total_time_ms = (time.time() - start_time) * 1000
        trace.success = True
        
        return QueryWithTraceResponse(
            success=True,
            response_text=trace.final_answer,
            session_id=session_id,
            thought_trace=trace.to_dict() if request.include_full_trace else {},
            specialist_used=specialist_used,
            delegation_reason=delegation_reason,
            tools_used=trace.tools_used,
            sources=result.sources if hasattr(result, 'sources') else [],
            metadata={
                "intent": analysis.intent.value if analysis.intent else None,
                "confidence": analysis.confidence,
                "total_time_ms": trace.total_time_ms,
                "step_count": len(trace.steps)
            }
        )
        
    except Exception as e:
        trace.success = False
        trace.total_time_ms = (time.time() - start_time) * 1000
        
        return QueryWithTraceResponse(
            success=False,
            response_text=f"Error processing query: {str(e)}",
            session_id=session_id,
            thought_trace=trace.to_dict() if request.include_full_trace else {},
            metadata={"error": str(e)}
        )


@router.get("/traces/{trace_id}")
async def get_trace_by_id(
    trace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific trace by ID with full details.
    
    Returns complete thought process including:
    - All reasoning steps
    - Tool calls and results
    - Delegation decisions
    - Timing information
    """
    tracer = get_tracer()
    trace = tracer.get_trace(trace_id)
    
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return {
        "trace": trace.to_dict(),
        "summary": {
            "total_steps": len(trace.spans),
            "duration_ms": trace.total_duration_ms,
            "tools_used": trace.total_tool_calls,
            "llm_calls": sum(1 for s in trace.spans if s.span_type.value == "llm"),
            "success": all(s.status.value != "error" for s in trace.spans)
        }
    }


@router.get("/specialists")
async def list_specialists(
    current_user: dict = Depends(get_current_user)
):
    """
    List all available specialized agents.
    
    Shows which specialists are available for delegation:
    - Technical Specifications Specialist
    - Compliance & Standards Specialist  
    - Troubleshooting Specialist
    - Pricing & Quotes Specialist
    """
    registry = get_specialist_registry()
    
    return {
        "specialists": registry.list_specialists(),
        "count": len(registry.specialists),
        "description": "Specialized agents provide more precise responses in their domain compared to the general agent."
    }


@router.post("/specialists/{domain}/query")
async def query_specialist_directly(
    domain: str,
    request: AgenticQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Query a specific specialist agent directly.
    
    Bypasses automatic routing and uses the specified specialist.
    Useful for testing or when you know which specialist is needed.
    """
    from src.agentic.agents.specialized_agents import SpecialistDomain
    
    registry = get_specialist_registry()
    
    # Validate domain
    try:
        specialist_domain = SpecialistDomain(domain)
    except ValueError:
        valid_domains = [d.value for d in SpecialistDomain]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid domain. Valid options: {valid_domains}"
        )
    
    specialist = registry.get_specialist(specialist_domain)
    if not specialist:
        raise HTTPException(status_code=404, detail=f"Specialist not found for domain: {domain}")
    
    import time
    start_time = time.time()
    
    result = await specialist.execute(
        query=request.query,
        context={
            **request.context,
            "user_role": current_user.get("role")
        }
    )
    
    return {
        "success": result.success,
        "response_text": result.final_answer,
        "specialist_used": domain,
        "trace": {
            "steps": [s.to_dict() for s in result.trace],
            "iterations": result.iterations,
            "tools_used": result.tools_used
        },
        "sources": result.sources,
        "metadata": {
            "total_time_ms": (time.time() - start_time) * 1000
        }
    }


# ============= SLM Training & Management Endpoints =============

class SLMTrainingRequest(BaseModel):
    """Request for training an SLM."""
    slm_type: str = Field(..., description="Type of SLM: intent_classifier, entity_extractor, product_matcher, compliance_checker")
    training_method: str = Field("sklearn", description="Training method: keyword, sklearn, transformer")
    source: str = Field("documents", description="Data source: documents, interactions, manual")
    document_filter: Optional[Dict[str, Any]] = Field(None, description="Filter for documents to use")
    num_examples: int = Field(100, description="Number of training examples to generate")
    hyperparams: Optional[Dict[str, Any]] = Field(None, description="Training hyperparameters")


class SLMPredictRequest(BaseModel):
    """Request for SLM prediction."""
    slm_type: str = Field(..., description="Type of SLM to use")
    text: str = Field(..., description="Text to classify/process")


@router.post("/slm/train")
async def train_slm(
    request: SLMTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Train an SLM on company data.
    
    This endpoint:
    1. Uses LLM to generate training data from company documents
    2. Trains a local SLM model (fast, privacy-preserving)
    3. The trained SLM can be used by specialist agents
    
    Admin only.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required for SLM training")
    
    from src.agentic.slm.training import (
        SLMType, SLMDataGenerator, SLMTrainer, TrainingDataset
    )
    
    # Validate SLM type
    try:
        slm_type = SLMType(request.slm_type)
    except ValueError:
        valid_types = [t.value for t in SLMType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SLM type. Valid options: {valid_types}"
        )
    
    # Get company documents
    documents = []
    
    if request.source == "documents":
        # Load documents from ChromaDB or data files
        try:
            import json
            from pathlib import Path
            
            # Try loading from scraped data
            data_path = Path("data/scraped_jd_jones.json")
            if data_path.exists():
                with open(data_path, "r", encoding="utf-8") as f:
                    scraped_data = json.load(f)
                    # Convert to document format
                    for item in scraped_data[:200]:  # Limit for training
                        if isinstance(item, dict):
                            documents.append({
                                "content": item.get("content", item.get("text", str(item))),
                                "metadata": item.get("metadata", {})
                            })
                        else:
                            documents.append({"content": str(item)})
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No training documents found. Please ingest data first."
        )
    
    # Generate training data
    orchestrator = get_orchestrator()
    generator = SLMDataGenerator(llm=orchestrator.reasoning_llm)
    
    # Create training dataset
    dataset = TrainingDataset(slm_type=slm_type)
    
    # Generate examples (this uses LLM to create training data)
    generated_examples = await generator.generate_from_documents(
        slm_type=slm_type,
        documents=documents,
        num_examples=request.num_examples
    )
    dataset.add_examples(generated_examples)
    
    # Train the model
    trainer = SLMTrainer()
    trained_model = await trainer.train(
        slm_type=slm_type,
        dataset=dataset,
        training_method=request.training_method,
        hyperparams=request.hyperparams
    )
    
    return {
        "success": True,
        "message": f"SLM {slm_type.value} trained successfully",
        "model": trained_model.to_dict(),
        "dataset_stats": dataset.get_statistics()
    }


@router.get("/slm/models")
async def list_slm_models(
    current_user: dict = Depends(get_current_user)
):
    """
    List all trained SLM models.
    
    Returns information about each trained model including:
    - Type and version
    - Training accuracy
    - Training date
    - Active status
    """
    from src.agentic.slm.training import SLMTrainer
    
    trainer = SLMTrainer()
    models = trainer.list_models()
    
    return {
        "models": [m.to_dict() for m in models],
        "count": len(models),
        "description": "SLMs are trained on company data for fast, local inference. The LLM remains the main brain."
    }


@router.post("/slm/predict")
async def slm_predict(
    request: SLMPredictRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Run inference using a trained SLM.
    
    This is much faster than LLM (< 10ms) for:
    - Intent classification
    - Entity extraction
    - Product matching
    - Compliance checking
    """
    from src.agentic.slm.training import SLMType, SLMInference
    
    try:
        slm_type = SLMType(request.slm_type)
    except ValueError:
        valid_types = [t.value for t in SLMType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SLM type. Valid options: {valid_types}"
        )
    
    inference = SLMInference()
    
    import time
    start_time = time.time()
    
    result = inference.predict(slm_type, request.text)
    
    result["inference_time_ms"] = (time.time() - start_time) * 1000
    result["slm_type"] = slm_type.value
    
    return result


@router.get("/slm/architecture")
async def get_slm_architecture(
    current_user: dict = Depends(get_current_user)
):
    """
    Get the SLM architecture documentation.
    
    Explains how LLM and SLMs work together:
    - LLM (GPT-4/Claude) = Main brain for orchestration and complex reasoning
    - SLMs = Fast local workers for classification, extraction, pre-filtering
    """
    return {
        "architecture": {
            "main_brain": {
                "type": "LLM (GPT-4/Claude)",
                "role": "Central orchestrator and complex reasoning",
                "capabilities": [
                    "Multi-step reasoning",
                    "Complex query understanding",
                    "Response synthesis",
                    "Tool orchestration"
                ],
                "latency": "1-5 seconds"
            },
            "slm_workers": {
                "type": "Small Language Models (trained on company data)",
                "role": "Fast local processing",
                "models": [
                    {
                        "name": "Intent Classifier",
                        "task": "Classify user query intent",
                        "latency": "< 10ms"
                    },
                    {
                        "name": "Entity Extractor",
                        "task": "Extract product codes, specs, standards",
                        "latency": "< 10ms"
                    },
                    {
                        "name": "Product Matcher",
                        "task": "Match queries to products",
                        "latency": "< 20ms"
                    },
                    {
                        "name": "Compliance Checker",
                        "task": "Identify compliance standards",
                        "latency": "< 10ms"
                    }
                ]
            },
            "workflow": [
                "1. Query arrives",
                "2. SLM classifies intent (< 10ms)",
                "3. SLM extracts entities (< 10ms)",
                "4. If simple: SLM handles directly",
                "5. If complex: Escalate to LLM brain",
                "6. LLM orchestrates tools and reasoning",
                "7. Response returned"
            ],
            "benefits": [
                "Fast response for common queries (< 50ms)",
                "Reduced LLM costs (70-80% queries handled by SLM)",
                "Privacy-preserving (SLMs run locally)",
                "Company-specific knowledge embedded in SLMs"
            ]
        }
    }


# ============= Customer Enquiry Endpoint =============

class EnquiryRequest(BaseModel):
    """Request for customer enquiry submission from external portal."""
    # Support both frontend naming conventions
    from_name: Optional[str] = Field(None, description="Customer name")
    from_email: Optional[str] = Field(None, description="Customer email")
    name: Optional[str] = Field(None, description="Customer name (alternative)")
    email: Optional[str] = Field(None, description="Customer email (alternative)")
    company: Optional[str] = Field(None, description="Company name")
    phone: Optional[str] = Field(None, description="Phone number")
    content: Optional[str] = Field(None, description="Enquiry content/message")
    message: Optional[str] = Field(None, description="Enquiry message (alternative)")
    enquiry_type: Optional[str] = Field("general", description="Type of enquiry")
    product_code: Optional[str] = Field(None, description="Related product code")
    urgency: Optional[str] = Field("normal", description="Urgency level")


class EnquiryResponse(BaseModel):
    """Response from customer enquiry submission - matches frontend expectations."""
    enquiry_id: str
    status: str = "pending"
    category: str = "general"
    response: str  # Acknowledgment message
    routed_to: Optional[str] = None
    estimated_response_time: Optional[str] = None


@router.post(
    "/enquiry",
    response_model=EnquiryResponse,
    summary="Submit customer enquiry",
    description="Submit a customer enquiry from the external portal. Enquiries are stored and routed to appropriate teams."
)
async def submit_enquiry(
    request: EnquiryRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a customer enquiry from the external portal.
    
    This endpoint receives enquiries from the EnquiryForm component
    and stores them in the admin enquiries store for internal team processing.
    """
    import uuid
    
    # Generate enquiry ID
    enquiry_id = f"ENQ-{uuid.uuid4().hex[:8].upper()}"
    
    # Get name and email (support both field naming conventions)
    customer_name = request.from_name or request.name or "Website Visitor"
    customer_email = request.from_email or request.email or ""
    message_content = request.message or request.content or ""
    
    # Determine routing based on enquiry type
    routing_map = {
        "product_info": "Sales Team",
        "technical": "Technical Support",
        "technical_support": "Technical Support",
        "quote_request": "Sales Team",
        "complaint": "Customer Service",
        "general": "Customer Service",
        "order_status": "Order Management",
        "documentation": "Technical Documentation",
        "other": "Customer Service"
    }
    routed_to = routing_map.get(request.enquiry_type, "Customer Service")
    
    # Determine priority based on urgency
    priority_map = {
        "urgent": "high",
        "normal": "medium",
        "low": "low"
    }
    priority = priority_map.get(request.urgency, "medium")
    
    # Build the enquiry data to store
    enquiry_data = {
        "enquiry_id": enquiry_id,
        "from_name": customer_name,
        "from_email": customer_email,
        "company": request.company or "",
        "content": message_content,
        "category": request.enquiry_type or "general",
        "status": "pending",
        "priority": priority,
        "routed_to": routed_to,
        "product_code": request.product_code or "",
        "source": "enquiry_form"
    }
    
    # Store in admin enquiries
    try:
        from src.api.routers.admin import _enquiries_store
        from datetime import datetime
        
        enquiry = {
            **enquiry_data,
            "created_at": datetime.now().isoformat(),
            "response": None
        }
        _enquiries_store.insert(0, enquiry)
        
    except Exception as e:
        import logging
        logging.warning(f"Failed to store enquiry in admin store: {e}")
    
    # Also save to enquiry models storage for the internal portal dashboard
    try:
        from src.enquiry.models import (
            Enquiry, CustomerDetails, EnquiryStatus, EnquiryPriority, EnquiryType, save_enquiry
        )
        
        # Map enquiry type to EnquiryType enum
        type_map = {
            "product_info": EnquiryType.PRODUCT_SELECTION,
            "technical": EnquiryType.TECHNICAL_ASSISTANCE,
            "technical_support": EnquiryType.TECHNICAL_ASSISTANCE,
            "quote_request": EnquiryType.QUOTATION,
            "complaint": EnquiryType.COMPLAINT,
            "general": EnquiryType.GENERAL,
            "order_status": EnquiryType.ORDER_STATUS,
            "documentation": EnquiryType.GENERAL,
            "other": EnquiryType.OTHER
        }
        
        # Map priority
        priority_enum_map = {
            "high": EnquiryPriority.HIGH,
            "medium": EnquiryPriority.MEDIUM,
            "low": EnquiryPriority.LOW,
            "urgent": EnquiryPriority.URGENT
        }
        
        customer = CustomerDetails(
            name=customer_name,
            email=customer_email,
            company=request.company or None,
            phone=None
        )
        
        # Include product code in message for AI analysis if provided
        full_message = message_content
        if request.product_code:
            full_message = f"[Product: {request.product_code}] {message_content}"
        
        enquiry_obj = Enquiry(
            id=enquiry_id,
            customer=customer,
            raw_message=full_message,
            subject=message_content[:100] + "..." if len(message_content) > 100 else message_content,
            enquiry_type=type_map.get(request.enquiry_type, EnquiryType.GENERAL),
            priority=priority_enum_map.get(priority, EnquiryPriority.MEDIUM),
            status=EnquiryStatus.NEW
        )
        
        save_enquiry(enquiry_obj)
        
        # Trigger background AI analysis for proper summarization
        from src.api.routers.enquiries import analyze_enquiry_background
        background_tasks.add_task(analyze_enquiry_background, enquiry_id)
        
    except Exception as e:
        import logging
        logging.warning(f"Failed to store enquiry in models storage: {e}")
    
    # Return response matching frontend expectations
    return EnquiryResponse(
        enquiry_id=enquiry_id,
        status="pending",
        category=request.enquiry_type or "general",
        response=f"Thank you for your enquiry. Your reference number is {enquiry_id}. Our {routed_to} will review your request and respond within 24 hours.",
        routed_to=routed_to,
        estimated_response_time="Within 24 hours"
    )
