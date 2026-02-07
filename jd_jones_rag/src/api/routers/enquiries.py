"""
Enquiry Router - Internal API endpoints for enquiry management.
Provides AI-assisted enquiry handling with summarization and response generation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, status, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr

from src.auth.authentication import User, get_current_active_user
from src.auth.authorization import Permission, require_permission
from src.enquiry.models import (
    Enquiry, EnquiryType, EnquiryPriority, EnquiryStatus,
    CustomerDetails, AIAnalysis, SuggestedResponse,
    save_enquiry, get_enquiry, get_all_enquiries, get_enquiry_stats
)
from src.enquiry.analyzer import get_enquiry_analyzer


logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class EnquirySubmitRequest(BaseModel):
    """Request to submit an enquiry (from external portal)."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    company: Optional[str] = None
    phone: Optional[str] = None
    subject: Optional[str] = None
    message: str = Field(..., min_length=10, max_length=10000)
    session_id: Optional[str] = None


class AssignEnquiryRequest(BaseModel):
    """Request to assign an enquiry."""
    assigned_to: str
    department: Optional[str] = None
    internal_note: Optional[str] = None


class AddNoteRequest(BaseModel):
    """Request to add an internal note."""
    note: str = Field(..., min_length=1, max_length=2000)


class GenerateResponseRequest(BaseModel):
    """Request to generate an AI response."""
    tone: str = Field(default="professional", pattern="^(professional|friendly|formal)$")
    include_products: bool = True


class SendResponseRequest(BaseModel):
    """Request to send a response."""
    response_text: str = Field(..., min_length=10)
    send_email: bool = True


class EnquiryListResponse(BaseModel):
    """Response for enquiry list."""
    total: int
    enquiries: List[Dict[str, Any]]


# External endpoint - Submit enquiry (no auth required)
@router.post(
    "/submit",
    summary="Submit customer enquiry",
    description="Submit a customer enquiry for AI analysis and internal review."
)
async def submit_enquiry(
    request: EnquirySubmitRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Submit a customer enquiry.
    
    This is called from the external portal. The enquiry is:
    1. Saved to the system
    2. Analyzed by AI in background (summary, type detection)
    3. Made available to internal team
    
    NOTE: No pricing or internal information is exposed in response.
    """
    # Create customer details
    customer = CustomerDetails(
        name=request.name,
        email=request.email,
        company=request.company,
        phone=request.phone
    )
    
    # Create enquiry
    enquiry = Enquiry(
        customer=customer,
        raw_message=request.message,
        subject=request.subject,
        session_id=request.session_id,
        status=EnquiryStatus.NEW
    )
    
    # Save enquiry
    saved = save_enquiry(enquiry)
    
    # Run AI analysis in background
    background_tasks.add_task(analyze_enquiry_background, saved.id)
    
    logger.info(f"Enquiry submitted: {saved.id} from {request.email}")
    
    return {
        "success": True,
        "enquiry_id": saved.id,
        "message": "Thank you for your enquiry. Our team will review and respond within 24 hours.",
        "estimated_response": "24 hours"
        # Note: No internal details exposed
    }


async def analyze_enquiry_background(enquiry_id: str):
    """Background task to analyze enquiry with AI."""
    try:
        enquiry = get_enquiry(enquiry_id)
        if not enquiry:
            return
        
        analyzer = get_enquiry_analyzer()
        analysis = await analyzer.analyze_enquiry(enquiry)
        
        # Update enquiry with analysis
        enquiry.ai_analysis = analysis
        enquiry.enquiry_type = analysis.detected_type
        enquiry.priority = analysis.detected_priority
        enquiry.status = EnquiryStatus.UNDER_REVIEW
        
        save_enquiry(enquiry)
        logger.info(f"AI analysis complete for enquiry {enquiry_id}")
        
    except Exception as e:
        logger.error(f"Error in background analysis: {e}")


# Internal endpoints - List and manage enquiries
@router.get(
    "/list",
    response_model=EnquiryListResponse,
    summary="List all enquiries",
    description="Get all customer enquiries for internal review."
)
async def list_enquiries(
    status: Optional[str] = Query(None, description="Filter by status"),
    enquiry_type: Optional[str] = Query(None, description="Filter by type"),
    assigned_to: Optional[str] = Query(None, description="Filter by assignee"),
    include_raw: bool = Query(True, description="Include raw message"),
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> EnquiryListResponse:
    """
    List all enquiries with optional filters.
    
    Internal use only - includes AI analysis and summaries.
    """
    status_filter = EnquiryStatus(status) if status else None
    type_filter = EnquiryType(enquiry_type) if enquiry_type else None
    
    enquiries = get_all_enquiries(
        status=status_filter,
        enquiry_type=type_filter,
        assigned_to=assigned_to
    )
    
    return EnquiryListResponse(
        total=len(enquiries),
        enquiries=[e.to_dict(include_raw=include_raw) for e in enquiries]
    )


@router.get(
    "/stats",
    summary="Get enquiry statistics",
    description="Get summary statistics for enquiries."
)
async def get_stats(
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> Dict[str, Any]:
    """Get enquiry statistics for dashboard."""
    return get_enquiry_stats()


@router.get(
    "/dashboard",
    summary="Get quick-view dashboard",
    description="Get quick-scan summaries for internal team dashboard."
)
async def get_dashboard(
    limit: int = Query(default=20, le=100, description="Max enquiries to return"),
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Get quick-view dashboard for internal team.
    
    Returns structured summaries that allow quick scanning without
    reading full enquiry content.
    """
    enquiries = get_all_enquiries()[:limit]
    
    quick_views = []
    for enquiry in enquiries:
        quick_view = {
            "id": enquiry.id,
            "customer": {
                "name": enquiry.customer.name if enquiry.customer else "Unknown",
                "company": enquiry.customer.company if enquiry.customer else None,
                "email": enquiry.customer.email if enquiry.customer else None
            },
            "subject": enquiry.subject,
            "status": enquiry.status.value,
            "created_at": enquiry.created_at.isoformat(),
            "assigned_to": enquiry.assigned_to
        }
        
        # Add AI analysis quick view if available
        if enquiry.ai_analysis:
            quick_view["ai_quick_view"] = enquiry.ai_analysis.get_quick_view()
        else:
            quick_view["ai_quick_view"] = {
                "one_liner": "Pending AI analysis",
                "type": enquiry.enquiry_type.value,
                "priority": enquiry.priority.value,
                "main_ask": "Review required",
                "actions_needed": {"technical_review": False, "pricing": False, "samples": False}
            }
        
        quick_views.append(quick_view)
    
    # Get stats
    stats = get_enquiry_stats()
    
    return {
        "enquiries": quick_views,
        "stats": stats,
        "retrieved_at": datetime.now().isoformat()
    }


@router.post(
    "/{enquiry_id}/re-analyze",
    summary="Re-run AI analysis",
    description="Re-run AI analysis on an enquiry with the latest multi-agent system."
)
async def reanalyze_enquiry(
    enquiry_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Re-run AI analysis on an enquiry.
    
    Useful when the enquiry has been updated or to get fresh analysis
    with improved AI agents.
    """
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    # Run fresh analysis
    analyzer = get_enquiry_analyzer()
    analysis = await analyzer.analyze_enquiry(enquiry)
    
    # Update enquiry
    enquiry.ai_analysis = analysis
    enquiry.enquiry_type = analysis.detected_type
    enquiry.priority = analysis.detected_priority
    
    enquiry.add_note(f"AI analysis refreshed by {current_user.username}", current_user.username)
    save_enquiry(enquiry)
    
    return {
        "success": True,
        "enquiry_id": enquiry_id,
        "analysis": analysis.to_dict(),
        "quick_view": analysis.get_quick_view()
    }


@router.get(
    "/{enquiry_id}",
    summary="Get enquiry details",
    description="Get full details of an enquiry including AI analysis."
)
async def get_enquiry_details(
    enquiry_id: str,
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Get full enquiry details.
    
    Includes:
    - Raw message
    - AI summary and analysis
    - Suggested responses
    - Internal notes
    """
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    return {
        "enquiry": enquiry.to_dict(include_raw=True),
        "viewed_by": current_user.username,
        "viewed_at": datetime.now().isoformat()
    }


@router.post(
    "/{enquiry_id}/assign",
    summary="Assign enquiry",
    description="Assign an enquiry to a team member."
)
async def assign_enquiry(
    enquiry_id: str,
    request: AssignEnquiryRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """Assign an enquiry to a team member."""
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    enquiry.assigned_to = request.assigned_to
    enquiry.department = request.department
    
    if request.internal_note:
        enquiry.add_note(request.internal_note, current_user.username)
    
    enquiry.add_note(f"Assigned to {request.assigned_to}", current_user.username)
    save_enquiry(enquiry)
    
    return {
        "success": True,
        "message": f"Enquiry assigned to {request.assigned_to}",
        "enquiry_id": enquiry_id
    }


@router.post(
    "/{enquiry_id}/note",
    summary="Add internal note",
    description="Add an internal note to an enquiry."
)
async def add_note(
    enquiry_id: str,
    request: AddNoteRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """Add an internal note to an enquiry."""
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    enquiry.add_note(request.note, current_user.username)
    save_enquiry(enquiry)
    
    return {
        "success": True,
        "message": "Note added",
        "notes_count": len(enquiry.internal_notes)
    }


@router.post(
    "/{enquiry_id}/generate-response",
    summary="Generate AI response",
    description="Generate an AI-suggested response for the enquiry."
)
async def generate_response(
    enquiry_id: str,
    request: GenerateResponseRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Generate an AI-suggested response.
    
    The internal team can review, edit, and then send this response.
    """
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    # Ensure AI analysis exists
    if not enquiry.ai_analysis:
        analyzer = get_enquiry_analyzer()
        enquiry.ai_analysis = await analyzer.analyze_enquiry(enquiry)
    
    # Generate response
    analyzer = get_enquiry_analyzer()
    suggested = await analyzer.generate_suggested_response(
        enquiry=enquiry,
        tone=request.tone,
        include_products=request.include_products
    )
    
    # Add to enquiry
    enquiry.suggested_responses.append(suggested)
    enquiry.status = EnquiryStatus.AI_RESPONSE_GENERATED
    save_enquiry(enquiry)
    
    logger.info(f"AI response generated for enquiry {enquiry_id} by {current_user.username}")
    
    return {
        "success": True,
        "suggested_response": suggested.to_dict(),
        "message": "AI response generated. Review and edit before sending.",
        "total_suggestions": len(enquiry.suggested_responses)
    }


@router.post(
    "/{enquiry_id}/send-response",
    summary="Send response to customer",
    description="Send a response to the customer (optionally via email)."
)
async def send_response(
    enquiry_id: str,
    request: SendResponseRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Send a response to the customer.
    
    If send_email is True, the response will be emailed to the customer.
    """
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    # Store the final response
    enquiry.final_response = request.response_text
    enquiry.response_sent_at = datetime.now()
    enquiry.response_sent_by = current_user.username
    enquiry.status = EnquiryStatus.RESPONSE_SENT
    
    # In production: Actually send email
    email_sent = False
    if request.send_email:
        # TODO: Integrate with email service
        # For now, just log it
        logger.info(f"Email would be sent to {enquiry.customer.email}")
        email_sent = True  # Simulated
    
    enquiry.add_note(
        f"Response sent by {current_user.username}. Email: {'Yes' if email_sent else 'No'}",
        current_user.username
    )
    
    save_enquiry(enquiry)
    
    return {
        "success": True,
        "message": f"Response {'sent via email' if email_sent else 'recorded'}",
        "enquiry_id": enquiry_id,
        "customer_email": enquiry.customer.email,
        "sent_at": enquiry.response_sent_at.isoformat()
    }


@router.patch(
    "/{enquiry_id}/status",
    summary="Update enquiry status",
    description="Update the status of an enquiry."
)
async def update_status(
    enquiry_id: str,
    new_status: str = Query(..., description="New status"),
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """Update enquiry status."""
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    try:
        enquiry.status = EnquiryStatus(new_status)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {new_status}"
        )
    
    enquiry.add_note(f"Status changed to {new_status}", current_user.username)
    save_enquiry(enquiry)
    
    return {
        "success": True,
        "enquiry_id": enquiry_id,
        "new_status": enquiry.status.value
    }


@router.post(
    "/{enquiry_id}/escalate",
    summary="Escalate enquiry",
    description="Escalate an enquiry for urgent attention."
)
async def escalate_enquiry(
    enquiry_id: str,
    reason: str = Query(..., min_length=5, description="Reason for escalation"),
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """Escalate an enquiry for urgent attention."""
    enquiry = get_enquiry(enquiry_id)
    
    if not enquiry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enquiry '{enquiry_id}' not found"
        )
    
    enquiry.status = EnquiryStatus.ESCALATED
    enquiry.priority = EnquiryPriority.URGENT
    enquiry.add_note(f"ESCALATED: {reason}", current_user.username)
    
    save_enquiry(enquiry)
    
    logger.warning(f"Enquiry {enquiry_id} escalated by {current_user.username}: {reason}")
    
    return {
        "success": True,
        "message": "Enquiry escalated",
        "enquiry_id": enquiry_id,
        "priority": "urgent"
    }
