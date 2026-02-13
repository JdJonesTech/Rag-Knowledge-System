"""
Quotation Router - Internal API endpoints for quotation management.
INTERNAL USE ONLY - Contains pricing and PDF generation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, status, Query, Depends, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.auth.authentication import User, get_current_active_user
from src.auth.authorization import Permission, require_permission
from src.quotation.models import (
    QuotationRequest, QuotationLineItem, CustomerInfo,
    QuotationStatus, get_quotation_request, get_all_quotation_requests,
    save_quotation_request, generate_quotation_number
)
from src.quotation.pdf_generator import get_pdf_generator
from src.quotation.analyzer import get_quotation_analyzer, QuotationAnalysis
from src.api.schemas.responses import (
    QuotationSubmitResponse, QuotationListResponse, QuotationDetailsResponse,
    QuotationUpdateResponse, QuotationSentResponse, QuotationStatsResponse,
    QuotationAnalysisResponse, QuotationDashboardResponse
)


logger = logging.getLogger(__name__)

router = APIRouter()

# Storage for AI analysis results (in-memory, use database in production)
_quotation_analyses: Dict[str, QuotationAnalysis] = {}


async def analyze_quotation_background(quotation_id: str):
    """Background task to analyze quotation with AI."""
    try:
        quotation = get_quotation_request(quotation_id)
        if not quotation:
            return
        
        analyzer = get_quotation_analyzer()
        analysis = await analyzer.analyze(quotation)
        
        # Store analysis
        _quotation_analyses[quotation_id] = analysis
        
        # Update quotation with AI summary
        quotation.ai_summary = analysis.one_liner
        quotation.ai_analysis = analysis
        save_quotation_request(quotation)
        
        logger.info(f"AI analysis complete for quotation {quotation_id}")
        
    except Exception as e:
        logger.error(f"Error in background quotation analysis: {e}")


class LineItemRequest(BaseModel):
    """Line item for quotation."""
    product_code: str
    product_name: str
    material_code: Optional[str] = None
    size: Optional[str] = None  # Human-readable size e.g., "12mm × 12mm"
    style: Optional[str] = None  # e.g., Braided, Die-formed, Wrapped
    material_grade: Optional[str] = None  # e.g., Standard, High Purity, Nuclear Grade
    colour: Optional[str] = None  # e.g., Natural/Grey, Black
    dimensions: Optional[Dict[str, Any]] = None  # {od, id, th}
    size_od: Optional[float] = None
    size_id: Optional[float] = None
    size_th: Optional[float] = None
    dimension_unit: str = "mm"  # Unit for dimensions: "mm" or "inch"
    rings_per_set: Optional[int] = None
    quantity: int = 1
    unit: str = "Nos."
    unit_price: Optional[float] = None  # Internal only
    specific_requirements: Optional[str] = None  # Detailed requirements
    notes: Optional[str] = None
    is_ai_suggested: bool = False  # Whether this item was AI-suggested



class UpdateQuotationRequest(BaseModel):
    """Request to update a quotation."""
    status: Optional[str] = None
    internal_notes: Optional[str] = None
    assigned_to: Optional[str] = None
    line_items: Optional[List[LineItemRequest]] = None


# Removed local QuotationListResponse model in favor of imported one


# Customer-facing request models
class CustomerInfoRequest(BaseModel):
    """Customer information for quotation request."""
    name: str
    company: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    designation: Optional[str] = None


class SpecificQuotationRequest(BaseModel):
    """
    Customer quotation request WITH specific requirements.
    Use this when customer provides detailed product/size/quantity info.
    """
    customer: CustomerInfoRequest
    reference_rfq: Optional[str] = None
    rfq_date: Optional[str] = None  # ISO format date
    industry: Optional[str] = None
    application: Optional[str] = None
    operating_conditions: Optional[str] = None
    special_requirements: Optional[str] = None
    line_items: List[LineItemRequest]  # Required - specific items


class GenericQuotationRequest(BaseModel):
    """
    Customer quotation request WITHOUT specific requirements.
    Use this when customer provides only a free-text message.
    AI system will suggest specifications for the selected product.
    """
    customer: CustomerInfoRequest
    message: str  # Free-text message describing their needs
    product_code: Optional[str] = None  # Product code from the selected product (e.g., NA 758H)
    product_name: Optional[str] = None  # Product name from the selected product
    reference_rfq: Optional[str] = None
    industry: Optional[str] = None
    attachments: Optional[List[str]] = None  # URLs or file references

# ==========================================
# CUSTOMER-FACING ENDPOINTS (External Portal)
# ==========================================

@router.post(
    "/external/submit-specific",
    summary="Submit quotation with specific requirements",
    description="Customer submits quotation request with detailed product requirements.",
    response_model=QuotationSubmitResponse
)
async def submit_specific_quotation(
    request: SpecificQuotationRequest,
    background_tasks: BackgroundTasks
) -> QuotationSubmitResponse:
    """
    Submit a quotation request WITH specific requirements.
    
    Use this when customer provides:
    - Specific product codes (NA 701, NA 710, etc.)
    - Quantities
    - Sizes (OD × ID × TH)
    - Application details
    
    This goes directly to pricing - NO AI processing needed.
    """
    from datetime import datetime
    
    # Create customer info
    customer = CustomerInfo(
        name=request.customer.name,
        company=request.customer.company,
        email=request.customer.email,
        phone=request.customer.phone,
        address=request.customer.address,
        designation=request.customer.designation
    )
    
    # Product catalog for auto-deriving material_code
    PRODUCT_CATALOG = {
        "NA 701": {"materials": ["graphite"]},
        "NA 702": {"materials": ["graphite", "ptfe"]},
        "NA 707": {"materials": ["ptfe"]},
        "NA 710": {"materials": ["ptfe", "graphite"]},
        "NA 715": {"materials": ["ptfe"]},
        "NA 750": {"materials": ["aramid"]},
        "NA 752": {"materials": ["aramid"]},
    }
    
    # Create line items with new fields
    line_items = []
    for item in request.line_items:
        # Build dimensions from explicit fields or from provided dimensions
        dimensions = item.dimensions
        if not dimensions and (item.size_od or item.size_id or item.size_th):
            dimensions = {
                "od": item.size_od,
                "id": item.size_id,
                "th": item.size_th
            }
        
        # Auto-derive material_code from product catalog if not provided
        material_code = item.material_code
        if not material_code and item.product_code:
            catalog_entry = PRODUCT_CATALOG.get(item.product_code.upper().strip(), {})
            catalog_materials = catalog_entry.get('materials', [])
            if catalog_materials:
                material_code = ', '.join(m.upper() for m in catalog_materials)
        
        line_items.append(QuotationLineItem(
            product_code=item.product_code,
            product_name=item.product_name,
            material_code=material_code,
            size=item.size,
            style=item.style,
            material_grade=item.material_grade,
            colour=item.colour,
            dimensions=dimensions,
            size_od=item.size_od,
            size_id=item.size_id,
            size_th=item.size_th,
            dimension_unit=getattr(item, 'dimension_unit', 'mm') or 'mm',
            rings_per_set=item.rings_per_set,
            quantity=item.quantity,
            unit=item.unit,
            specific_requirements=item.specific_requirements,
            notes=item.notes,
            is_ai_suggested=item.is_ai_suggested
        ))
    
    # Create quotation request
    quotation = QuotationRequest(
        customer=customer,
        reference_rfq=request.reference_rfq,
        rfq_date=datetime.fromisoformat(request.rfq_date) if request.rfq_date else None,
        industry=request.industry,
        application=request.application,
        operating_conditions=request.operating_conditions,
        special_requirements=request.special_requirements,
        line_items=line_items,
        status=QuotationStatus.PENDING,
        requires_ai_processing=False  # Customer provided specific requirements
    )
    
    saved = save_quotation_request(quotation)
    logger.info(f"Specific quotation request submitted: {saved.id} from {customer.company}")
    
    # Trigger background AI analysis for better summaries
    background_tasks.add_task(analyze_quotation_background, saved.id)
    
    return {
        "success": True,
        "request_id": saved.id,
        "message": "Thank you! Your quotation request has been received. Our team will review and respond shortly.",
        "type": "specific",
        "items_count": len(line_items),
        "status": saved.status.value
    }


@router.post(
    "/external/submit-generic",
    summary="Submit generic quotation request",
    description="Customer submits quotation request without specific details. AI will process.",
    response_model=QuotationSubmitResponse
)
async def submit_generic_quotation(
    request: GenericQuotationRequest,
    background_tasks: BackgroundTasks
) -> QuotationSubmitResponse:
    """
    Submit a quotation request WITHOUT specific requirements.
    
    Use this when customer provides:
    - Only a free-text message describing their needs
    - No specific product codes or quantities
    
    AI system will:
    1. Extract requirements from the message
    2. Identify suitable products
    3. Recommend specifications
    4. Flag for internal team review
    """
    from datetime import datetime
    
    # Create customer info
    customer = CustomerInfo(
        name=request.customer.name,
        company=request.customer.company,
        email=request.customer.email,
        phone=request.customer.phone,
        address=request.customer.address,
        designation=request.customer.designation
    )
    
    # Get product code - prioritize from request, then extract from message
    import re
    product_codes = []
    
    # First: use product_code from the request (passed from the selected product)
    if request.product_code:
        # Normalize: "NA758H" -> "NA 758H"
        code = request.product_code.upper().strip()
        code = re.sub(r'NA(\d)', r'NA \1', code)
        product_codes.append(code)
        logger.info(f"Using product code from request: {code}")
    else:
        # Fallback: extract product codes from the message using regex
        extracted = re.findall(r'NA\s*\d{3}[A-Z]*', request.message, re.IGNORECASE)
        extracted = list(set([c.upper().strip() for c in extracted]))
        # Normalize: "NA758H" -> "NA 758H"
        product_codes = [re.sub(r'NA(\d)', r'NA \1', code) for code in extracted]
        logger.info(f"Extracted product codes from message: {product_codes}")
    
    # Product catalog for auto-deriving material_code
    PRODUCT_CATALOG = {
        "NA 701": {"materials": ["graphite"]},
        "NA 702": {"materials": ["graphite", "ptfe"]},
        "NA 707": {"materials": ["ptfe"]},
        "NA 710": {"materials": ["ptfe", "graphite"]},
        "NA 715": {"materials": ["ptfe"]},
        "NA 750": {"materials": ["aramid"]},
        "NA 752": {"materials": ["aramid"]},
    }
    
    # Create line items from product codes
    from src.quotation.models import QuotationLineItem
    line_items = []
    for code in product_codes[:5]:  # Max 5 products
        # Auto-derive material_code from product catalog
        catalog_entry = PRODUCT_CATALOG.get(code.upper().strip(), {})
        catalog_materials = catalog_entry.get('materials', [])
        material_code = ', '.join(m.upper() for m in catalog_materials) if catalog_materials else None
        
        line_items.append(QuotationLineItem(
            product_code=code,
            product_name=request.product_name or code,
            material_code=material_code,
            quantity=1,
            is_ai_suggested=False
        ))
    
    # Create quotation request with line items
    quotation = QuotationRequest(
        customer=customer,
        reference_rfq=request.reference_rfq,
        industry=request.industry,
        line_items=line_items,  # Populated from selected product
        status=QuotationStatus.PENDING,
        requires_ai_processing=True,  # Needs AI to suggest specifications
        original_message=request.message  # Store original customer message
    )
    
    saved = save_quotation_request(quotation)
    logger.info(f"Generic quotation request submitted: {saved.id} from {customer.company} with products: {product_codes}")
    
    # Trigger AI analysis to get specification recommendations
    try:
        analyzer = get_quotation_analyzer()
        analysis = await analyzer.analyze(saved)
        
        # Store analysis in both places for compatibility
        _quotation_analyses[saved.id] = analysis
        saved.ai_analysis = analysis  # Store on the request object itself
        saved.ai_processed = True
        
        # Replace line items with AI-suggested line items (with specs filled in)
        if hasattr(analysis, 'sub_agent_results') and analysis.sub_agent_results:
            ai_suggested_items = analysis.sub_agent_results.get('ai_suggested_line_items', [])
            if ai_suggested_items:
                # Convert dict items back to QuotationLineItem objects
                saved.line_items = []
                for item_data in ai_suggested_items:
                    if isinstance(item_data, dict):
                        # Auto-derive material_code if not set by AI
                        ai_material_code = item_data.get('material_code')
                        if not ai_material_code:
                            ai_code = (item_data.get('product_code') or '').upper().strip()
                            ai_cat = PRODUCT_CATALOG.get(ai_code, {})
                            ai_mats = ai_cat.get('materials', [])
                            if ai_mats:
                                ai_material_code = ', '.join(m.upper() for m in ai_mats)
                        
                        ai_item = QuotationLineItem(
                            product_code=item_data.get('product_code', ''),
                            product_name=item_data.get('product_name', ''),
                            size=item_data.get('size'),
                            size_od=item_data.get('size_od'),
                            size_id=item_data.get('size_id'),
                            size_th=item_data.get('size_th'),
                            dimension_unit=item_data.get('dimension_unit', 'mm'),
                            style=item_data.get('style'),
                            material_grade=item_data.get('material_grade'),
                            material_code=ai_material_code,
                            colour=item_data.get('colour'),
                            quantity=item_data.get('quantity', 1),
                            unit=item_data.get('unit', 'Nos.'),
                            rings_per_set=item_data.get('rings_per_set'),
                            dimensions=item_data.get('dimensions'),
                            specific_requirements=item_data.get('specific_requirements'),
                            is_ai_suggested=True,
                            ai_confidence=item_data.get('ai_confidence', 0.75),
                            notes=item_data.get('notes')
                        )
                        saved.line_items.append(ai_item)
                logger.info(f"Updated quotation with {len(saved.line_items)} AI-suggested line items with specs")
        
        save_quotation_request(saved)
        
        ai_summary = analysis.get_quick_view()
    except Exception as e:
        logger.warning(f"AI analysis failed for {saved.id}: {e}")
        import traceback
        traceback.print_exc()
        ai_summary = None
    
    return {
        "success": True,
        "request_id": saved.id,
        "message": "Thank you! Your request has been received. Our AI system is analyzing your requirements, and our team will prepare a detailed quotation shortly.",
        "type": "generic",
        "ai_processing": True,
        "ai_summary": ai_summary,
        "status": saved.status.value,
        "extracted_products": product_codes
    }


# ==========================================
# INTERNAL ENDPOINTS (Requires Authentication)
# ==========================================

# Endpoints
@router.get(
    "/quotations",
    response_model=QuotationListResponse,
    summary="List all quotation requests",
    description="Get all quotation requests for internal review."
)
async def list_quotations(
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> QuotationListResponse:
    """
    List all quotation requests.
    
    Internal use only - includes full details and pricing.
    """
    status_filter = QuotationStatus(status) if status else None
    quotations = get_all_quotation_requests(status_filter)
    
    return QuotationListResponse(
        total=len(quotations),
        quotations=[q.to_dict(include_internal=True) for q in quotations]
    )


@router.get(
    "/quotations/{request_id}",
    summary="Get quotation details",
    description="Get full details of a quotation request.",
    response_model=QuotationDetailsResponse
)
async def get_quotation(
    request_id: str,
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> QuotationDetailsResponse:
    """
    Get full quotation request details.
    
    Internal use only - includes pricing.
    """
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    return {
        "quotation": quotation.to_dict(include_internal=True),
        "retrieved_by": current_user.username,
        "retrieved_at": datetime.now().isoformat()
    }


@router.put(
    "/quotations/{request_id}",
    summary="Update quotation",
    description="Update a quotation request with pricing and status.",
    response_model=QuotationUpdateResponse
)
async def update_quotation(
    request_id: str,
    update: UpdateQuotationRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> QuotationUpdateResponse:
    """
    Update a quotation request.
    
    Used by internal team to add pricing, update status, assign to sales rep.
    """
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    # Update status
    if update.status:
        quotation.status = QuotationStatus(update.status)
        if update.status == "quoted" and not quotation.quotation_number:
            quotation.quotation_number = generate_quotation_number()
            quotation.quoted_at = datetime.now()
    
    # Update internal notes
    if update.internal_notes is not None:
        quotation.internal_notes = update.internal_notes
    
    # Update assigned to
    if update.assigned_to is not None:
        quotation.assigned_to = update.assigned_to
    
    # Update line items with pricing
    if update.line_items:
        quotation.line_items = []
        total = 0.0
        for item in update.line_items:
            line_item = QuotationLineItem(
                product_code=item.product_code,
                product_name=item.product_name,
                material_code=item.material_code,
                size_od=item.size_od,
                size_id=item.size_id,
                size_th=item.size_th,
                dimension_unit=getattr(item, 'dimension_unit', 'mm') or 'mm',
                rings_per_set=item.rings_per_set,
                quantity=item.quantity,
                unit=item.unit,
                unit_price=item.unit_price,
                amount=(item.unit_price or 0) * item.quantity,
                notes=item.notes
            )
            quotation.line_items.append(line_item)
            total += line_item.amount or 0
        
        quotation.total_amount = total
        quotation.gst_amount = total * 0.18  # 18% GST
        quotation.grand_total = total + quotation.gst_amount
    
    # Save updated quotation
    saved = save_quotation_request(quotation)
    
    logger.info(f"Quotation {request_id} updated by {current_user.username}")
    
    return {
        "success": True,
        "quotation": saved.to_dict(include_internal=True),
        "updated_by": current_user.username,
        "updated_at": datetime.now().isoformat()
    }


@router.post(
    "/quotations/{request_id}/generate-pdf",
    summary="Generate quotation PDF",
    description="Generate a formal quotation PDF document."
)
async def generate_pdf(
    request_id: str,
    include_pricing: bool = Query(True, description="Include pricing in PDF"),
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Response:
    """
    Generate a quotation PDF.
    
    IMPORTANT: This is for INTERNAL USE ONLY.
    - With pricing: For sending to approved customers
    - Without pricing: For internal review/preview
    
    Pricing PDFs should NEVER be shared via external portal.
    """
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    # Ensure quotation number is assigned
    if not quotation.quotation_number:
        quotation.quotation_number = generate_quotation_number()
        quotation.quoted_at = datetime.now()
        quotation.status = QuotationStatus.QUOTED
        save_quotation_request(quotation)
    
    # Generate PDF
    pdf_generator = get_pdf_generator()
    pdf_bytes = pdf_generator.generate_quotation_pdf(
        request=quotation,
        include_pricing=include_pricing
    )
    
    logger.info(f"PDF generated for quotation {quotation.quotation_number} by {current_user.username}")
    
    filename = f"Quotation_{quotation.quotation_number}.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.post(
    "/quotations/{request_id}/mark-sent",
    summary="Mark quotation as sent",
    description="Mark a quotation as sent to customer.",
    response_model=QuotationSentResponse
)
async def mark_sent(
    request_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> QuotationSentResponse:
    """Mark quotation as sent to customer."""
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    quotation.status = QuotationStatus.SENT
    save_quotation_request(quotation)
    
    return {
        "success": True,
        "quotation_number": quotation.quotation_number,
        "status": quotation.status.value,
        "message": "Quotation marked as sent"
    }


@router.get(
    "/quotations/stats/summary",
    summary="Get quotation statistics",
    description="Get summary statistics for quotations.",
    response_model=QuotationStatsResponse
)
async def get_stats(
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> QuotationStatsResponse:
    """Get quotation statistics for dashboard."""
    all_quotations = get_all_quotation_requests()
    
    stats = {
        "total": len(all_quotations),
        "by_status": {},
        "recent": []
    }
    
    for status in QuotationStatus:
        count = len([q for q in all_quotations if q.status == status])
        stats["by_status"][status.value] = count
    
    # Recent 5 quotations
    stats["recent"] = [
        {
            "id": q.id,
            "customer": q.customer.company if q.customer else "Unknown",
            "status": q.status.value,
            "created_at": q.created_at.isoformat()
        }
        for q in all_quotations[:5]
    ]
    
    return stats


# Storage for AI analysis results
_quotation_analyses: Dict[str, QuotationAnalysis] = {}


@router.post(
    "/quotations/{request_id}/analyze",
    summary="Analyze quotation with AI",
    description="Run multi-agent AI analysis on a quotation request.",
    response_model=QuotationAnalysisResponse
)
async def analyze_quotation(
    request_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> QuotationAnalysisResponse:
    """
    Analyze a quotation request using multi-agent AI system.
    
    Sub-agents:
    1. RequirementsAnalyzer - Extract technical requirements
    2. ProductMatcher - Match requirements to products
    3. PricingEstimator - Estimate pricing
    4. DeliveryEstimator - Estimate delivery timeline
    5. SummaryGenerator - Generate structured summary
    """
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    try:
        analyzer = get_quotation_analyzer()
        analysis = await analyzer.analyze(quotation)
        
        # Store analysis
        _quotation_analyses[request_id] = analysis
        
        logger.info(f"AI analysis completed for quotation {request_id}")
        
        return {
            "success": True,
            "quotation_id": request_id,
            "analysis": analysis.to_dict(),
            "quick_view": analysis.get_quick_view(),
            "analyzed_by": current_user.username,
            "analyzed_at": analysis.analyzed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI analysis failed for quotation {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {str(e)}"
        )


@router.post(
    "/quotations/{request_id}/re-analyze",
    summary="Re-run AI analysis",
    description="Re-run multi-agent AI analysis on a quotation request.",
    response_model=QuotationAnalysisResponse
)
async def re_analyze_quotation(
    request_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> QuotationAnalysisResponse:
    """Re-run AI analysis on an existing quotation."""
    return await analyze_quotation(request_id, current_user)


@router.get(
    "/quotations/{request_id}/analysis",
    summary="Get quotation analysis",
    description="Get the AI analysis for a quotation request.",
    response_model=QuotationAnalysisResponse
)
async def get_quotation_analysis(
    request_id: str,
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> QuotationAnalysisResponse:
    """Get the AI analysis for a quotation."""
    quotation = get_quotation_request(request_id)
    
    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation request '{request_id}' not found"
        )
    
    analysis = _quotation_analyses.get(request_id)
    
    if not analysis:
        return {
            "quotation_id": request_id,
            "has_analysis": False,
            "message": "No AI analysis available. Run /analyze first."
        }
    
    return {
        "quotation_id": request_id,
        "has_analysis": True,
        "analysis": analysis.to_dict(),
        "quick_view": analysis.get_quick_view()
    }


@router.get(
    "/quotations/dashboard/overview",
    summary="Quotation dashboard",
    description="Get dashboard overview with AI quick-view summaries.",
    response_model=QuotationDashboardResponse
)
async def get_dashboard(
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> QuotationDashboardResponse:
    """
    Get quotation dashboard with AI-powered quick summaries.
    
    Returns:
    - Statistics by status
    - Quick-view summaries for each quotation
    - Pending actions and recommendations
    """
    all_quotations = get_all_quotation_requests()
    
    # Build dashboard data
    quotations_with_analysis = []
    for q in all_quotations:
        analysis = _quotation_analyses.get(q.id)
        entry = {
            "id": q.id,
            "quotation_number": q.quotation_number,
            "customer": {
                "name": q.customer.name if q.customer else "Unknown",
                "company": q.customer.company if q.customer else "Unknown",
                "email": q.customer.email if q.customer else ""
            },
            "status": q.status.value,
            "created_at": q.created_at.isoformat(),
            "line_items_count": len(q.line_items),
            "assigned_to": q.assigned_to
        }
        
        if analysis:
            entry["ai_quick_view"] = analysis.get_quick_view()
            entry["ai_analysis"] = analysis.to_dict()
        else:
            # Generate basic summary without full analysis
            entry["ai_quick_view"] = {
                "one_liner": f"Quotation for {len(q.line_items)} items from {q.customer.company if q.customer else 'Unknown'}",
                "priority": "medium",
                "complexity": "standard",
                "products": [item.product_code for item in q.line_items[:3]],
                "estimated_value": None,
                "delivery_days": None,
                "actions_needed": {
                    "engineering_review": False,
                    "custom_pricing": True,  # Always needs pricing
                    "sample": False
                }
            }
        
        quotations_with_analysis.append(entry)
    
    # Calculate stats
    stats = {
        "total": len(all_quotations),
        "by_status": {},
        "pending_pricing": len([q for q in all_quotations if q.status == QuotationStatus.PENDING]),
        "ready_to_send": len([q for q in all_quotations if q.status == QuotationStatus.QUOTED]),
        "total_value_pending": sum(
            q.grand_total or 0 
            for q in all_quotations 
            if q.status in [QuotationStatus.PENDING, QuotationStatus.IN_REVIEW, QuotationStatus.QUOTED]
        )
    }
    
    for status_enum in QuotationStatus:
        stats["by_status"][status_enum.value] = len([q for q in all_quotations if q.status == status_enum])
    
    return {
        "quotations": quotations_with_analysis,
        "stats": stats,
        "retrieved_at": datetime.now().isoformat()
    }
