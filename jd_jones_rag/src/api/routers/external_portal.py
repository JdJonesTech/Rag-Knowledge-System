"""
External Portal Router - Customer-Facing API Endpoints
Handles customer interactions through decision tree and form submissions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field, EmailStr

from src.external_system.classifier import IntentClassifier, CustomerIntent
from src.external_system.decision_tree import DecisionTree, NodeType
from src.external_system.response_generator import ResponseGenerator
from src.agentic.memory.conversation_memory import ConversationMemory
from src.api.schemas.responses import (
    DecisionTreeResponse, DecisionTreeNodeResponse, NavigationResponse,
    ClassificationResponse, QueryProcessResponse, FormSubmissionResponse,
    ContactFormResponse, QuoteRequestResponse, OrderTrackingResponse,
    FAQResponse, SessionResponse
)


logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
intent_classifier = IntentClassifier(use_llm=True)
decision_tree = DecisionTree()
response_generator = ResponseGenerator()

# Initialize conversation memory for short-term context
conversation_memory = ConversationMemory(
    max_messages=50,
    context_window=10,
    ttl_hours=24
)

# In-memory session storage (use Redis in production)
customer_sessions: Dict[str, Dict[str, Any]] = {}


# Request/Response Models
class CustomerQueryRequest(BaseModel):
    """Customer query request."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "I need information about your industrial equipment",
                "session_id": None
            }
        }


class NavigationRequest(BaseModel):
    """Decision tree navigation request."""
    node_id: str
    option_index: Optional[int] = None
    collected_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class FormSubmissionRequest(BaseModel):
    """Form submission request."""
    node_id: str
    form_data: Dict[str, Any]
    session_id: Optional[str] = None


class ContactFormRequest(BaseModel):
    """Contact form request."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)
    form_type: str = "general"


class QuoteRequestForm(BaseModel):
    """Quote request form - Enhanced with structured product requests."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=5, max_length=20)
    company: str = Field(..., min_length=1, max_length=200)
    designation: Optional[str] = None
    address: Optional[str] = None
    
    # RFQ Reference
    rfq_number: Optional[str] = None
    rfq_date: Optional[str] = None
    due_date: Optional[str] = None
    
    # Requirements
    industry: Optional[str] = None
    application: Optional[str] = None
    operating_conditions: Optional[str] = None
    special_requirements: Optional[str] = None
    
    # Products - can be text description or structured
    products: Optional[str] = None  # Free text description
    product_items: Optional[List[Dict[str, Any]]] = None  # Structured items
    
    quantity: Optional[int] = None
    delivery_date: Optional[str] = None
    notes: Optional[str] = None


class OrderTrackingRequest(BaseModel):
    """Order tracking request."""
    order_number: str
    email: EmailStr


# Helper functions
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in customer_sessions:
        return session_id
    
    new_session_id = f"cust_{uuid.uuid4().hex[:12]}"
    customer_sessions[new_session_id] = {
        "created_at": datetime.now().isoformat(),
        "current_node": "root",
        "collected_data": {},
        "history": []
    }
    return new_session_id


# Endpoints
@router.get(
    "/decision-tree",
    summary="Get decision tree structure",
    description="Get the complete decision tree structure for the customer portal.",
    response_model=DecisionTreeResponse
)
async def get_decision_tree() -> DecisionTreeResponse:
    """Get the complete decision tree."""
    return {
        "root_node": decision_tree.get_root().to_dict(),
        "tree_structure": decision_tree.get_tree_structure()
    }


@router.get(
    "/decision-tree/node/{node_id}",
    summary="Get specific node",
    description="Get a specific node from the decision tree.",
    response_model=DecisionTreeNodeResponse
)
async def get_node(node_id: str) -> DecisionTreeNodeResponse:
    """Get a specific decision tree node."""
    node = decision_tree.get_node(node_id)
    
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{node_id}' not found"
        )
    
    return node.to_dict()


@router.post(
    "/navigate",
    summary="Navigate decision tree",
    description="Navigate to the next node in the decision tree.",
    response_model=NavigationResponse
)
async def navigate(request: NavigationRequest) -> NavigationResponse:
    """
    Navigate through the decision tree.
    
    Args:
        request: Navigation request with node and option
        
    Returns:
        Next node and session information
    """
    session_id = get_or_create_session(request.session_id)
    session = customer_sessions[session_id]
    
    # Get current node
    current_node = decision_tree.get_node(request.node_id)
    if not current_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{request.node_id}' not found"
        )
    
    # Store collected data
    if request.collected_data:
        session["collected_data"].update(request.collected_data)
    
    # Navigate to next node if option provided
    next_node = current_node
    if request.option_index is not None:
        next_node = decision_tree.navigate(request.node_id, request.option_index)
        if not next_node:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid navigation option"
            )
    
    # Update session
    session["current_node"] = next_node.node_id
    session["history"].append({
        "from": request.node_id,
        "to": next_node.node_id,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "session_id": session_id,
        "current_node": next_node.to_dict(),
        "collected_data": session["collected_data"],
        "history_length": len(session["history"])
    }


@router.post(
    "/classify",
    summary="Classify customer intent",
    description="Classify a customer query to determine intent.",
    response_model=ClassificationResponse
)
async def classify_query(request: CustomerQueryRequest) -> ClassificationResponse:
    """
    Classify customer query intent.
    
    Args:
        request: Customer query
        
    Returns:
        Classification result with suggested starting node
    """
    session_id = get_or_create_session(request.session_id)
    
    # Classify the query
    classification = await intent_classifier.classify(request.query)
    
    # Get appropriate starting node
    starting_node = decision_tree.get_node_for_intent(classification.primary_intent)
    
    return {
        "session_id": session_id,
        "classification": classification.to_dict(),
        "suggested_node": starting_node.to_dict(),
        "intent_description": intent_classifier.get_intent_description(
            classification.primary_intent
        )
    }


@router.post(
    "/query",
    summary="Process customer query",
    description="Process a natural language customer query and generate a response.",
    response_model=QueryProcessResponse
)
async def process_query(request: CustomerQueryRequest) -> QueryProcessResponse:
    """
    Process customer query with classification and response generation.
    Uses conversation memory for context tracking.
    
    Args:
        request: Customer query
        
    Returns:
        Generated response with classification and suggestions
    """
    session_id = get_or_create_session(request.session_id)
    
    # Store user message in short-term memory
    conversation_memory.add_message(
        session_id=session_id,
        role="user",
        content=request.query
    )
    
    # Get conversation history for context
    conversation_context = conversation_memory.get_context(session_id, num_messages=5)
    conversation_history = []
    if conversation_context and conversation_context.messages:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in conversation_context.messages[:-1]  # Exclude current message
        ]
    
    # Classify intent
    classification = await intent_classifier.classify(request.query)
    
    # Build enhanced session context with conversation history
    session_context = customer_sessions.get(session_id, {}).get("collected_data", {})
    if conversation_history:
        session_context["conversation_history"] = conversation_history
        session_context["conversation_summary"] = conversation_context.summary if conversation_context else ""
    
    # Generate response with conversation context
    response = await response_generator.generate_response(
        query=request.query,
        classification=classification,
        session_context=session_context
    )
    
    # Store assistant response in short-term memory
    conversation_memory.add_message(
        session_id=session_id,
        role="assistant",
        content=response["response"],
        metadata={
            "intent": response["intent"],
            "confidence": response["confidence"],
            "sources": response.get("sources", [])
        }
    )
    
    # Get suggested node
    suggested_node = decision_tree.get_node_for_intent(classification.primary_intent)
    
    return {
        "session_id": session_id,
        "response": response["response"],
        "intent": response["intent"],
        "confidence": response["confidence"],
        "sources": response["sources"],
        "suggested_actions": response["suggested_actions"],
        "suggested_node": suggested_node.to_dict(),
        "conversation_turn": len(conversation_context.messages) if conversation_context else 1
    }



@router.post(
    "/submit-form",
    summary="Submit form data",
    description="Submit form data collected in the decision tree.",
    response_model=FormSubmissionResponse
)
async def submit_form(request: FormSubmissionRequest) -> FormSubmissionResponse:
    """
    Submit form data from a decision tree node.
    
    Args:
        request: Form submission with data
        
    Returns:
        Submission confirmation
    """
    session_id = get_or_create_session(request.session_id)
    
    # Get the form node
    node = decision_tree.get_node(request.node_id)
    if not node or node.node_type != NodeType.FORM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid form node"
        )
    
    # Validate required fields
    for field in node.form_fields:
        if field.required and field.name not in request.form_data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required field: {field.label}"
            )
    
    # Store form data
    submission_id = f"sub_{uuid.uuid4().hex[:8]}"
    
    # In production, this would:
    # 1. Store in database
    # 2. Send notification emails
    # 3. Create tickets/quotes as needed
    # 4. Integrate with CRM
    
    submission = {
        "submission_id": submission_id,
        "session_id": session_id,
        "node_id": request.node_id,
        "action_type": node.action_type,
        "form_data": request.form_data,
        "submitted_at": datetime.now().isoformat()
    }
    
    # Update session
    session = customer_sessions.get(session_id, {})
    if "submissions" not in session:
        session["submissions"] = []
    session["submissions"].append(submission)
    
    return {
        "success": True,
        "submission_id": submission_id,
        "message": "Form submitted successfully",
        "action_type": node.action_type,
        "next_steps": "Our team will review your submission and contact you shortly."
    }


@router.post(
    "/contact",
    summary="Submit contact form",
    description="Submit a general contact form.",
    response_model=ContactFormResponse
)
async def submit_contact_form(form: ContactFormRequest) -> ContactFormResponse:
    """
    Submit a contact form.
    
    Args:
        form: Contact form data
        
    Returns:
        Submission confirmation
    """
    submission_id = f"contact_{uuid.uuid4().hex[:8]}"
    
    # In production: store in database, send emails, create CRM ticket
    
    return {
        "success": True,
        "submission_id": submission_id,
        "message": "Thank you for contacting us. We'll respond within 1 business day.",
        "contact_info": {
            "name": form.name,
            "email": form.email
        }
    }


@router.post(
    "/quote-request",
    summary="Submit quote request",
    description="Submit a product quote request. This creates a quotation request for internal review.",
    response_model=QuoteRequestResponse
)
async def submit_quote_request(form: QuoteRequestForm) -> QuoteRequestResponse:
    """
    Submit a quote request.
    
    Creates a quotation request that is saved for internal review.
    The internal team can then generate a formal quotation PDF.
    
    NOTE: Pricing is NEVER exposed in this endpoint - only in internal portal.
    
    Args:
        form: Quote request data
        
    Returns:
        Quote confirmation with reference number
    """
    from datetime import datetime
    from src.quotation.models import (
        QuotationRequest, QuotationLineItem, CustomerInfo,
        QuotationStatus, save_quotation_request
    )
    
    # Create customer info
    customer = CustomerInfo(
        name=form.name,
        company=form.company,
        email=form.email,
        phone=form.phone,
        address=form.address,
        designation=form.designation
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
    
    # Create line items from structured products or parse from text
    line_items = []
    if form.product_items:
        for item in form.product_items:
            # Auto-derive material_code from product catalog if not provided
            material_code = item.get("material_code")
            if not material_code:
                product_code = (item.get("product_code") or "").upper().strip()
                catalog_entry = PRODUCT_CATALOG.get(product_code, {})
                catalog_materials = catalog_entry.get('materials', [])
                if catalog_materials:
                    material_code = ', '.join(m.upper() for m in catalog_materials)
            
            line_items.append(QuotationLineItem(
                product_code=item.get("product_code", "TBD"),
                product_name=item.get("product_name", "Product"),
                size=item.get("size"),
                size_od=item.get("size_od"),
                size_id=item.get("size_id"),
                size_th=item.get("size_th"),
                dimension_unit=item.get("dimension_unit", "mm"),
                style=item.get("style"),
                material_grade=item.get("material_grade"),
                material_code=material_code,
                colour=item.get("colour"),
                quantity=item.get("quantity", 1),
                unit=item.get("unit", "Nos."),
                rings_per_set=item.get("rings_per_set"),
                specific_requirements=item.get("specific_requirements"),
                notes=item.get("notes"),
                is_ai_suggested=item.get("is_ai_suggested", False)
            ))
    elif form.products:
        # Create a single line item for text-based request
        line_items.append(QuotationLineItem(
            product_code="TBD",
            product_name="To be determined based on requirements",
            quantity=form.quantity or 1,
            notes=form.products
        ))
    
    # Parse dates
    rfq_date = None
    if form.rfq_date:
        try:
            rfq_date = datetime.fromisoformat(form.rfq_date)
        except ValueError:
            pass
    
    due_date = None
    if form.due_date:
        try:
            due_date = datetime.fromisoformat(form.due_date)
        except ValueError:
            pass
    
    # Create quotation request
    quote_request = QuotationRequest(
        customer=customer,
        reference_rfq=form.rfq_number,
        rfq_date=rfq_date,
        due_date=due_date,
        industry=form.industry,
        application=form.application,
        operating_conditions=form.operating_conditions,
        special_requirements=form.special_requirements or form.notes,
        line_items=line_items,
        status=QuotationStatus.PENDING
    )
    
    # Save the request
    saved_request = save_quotation_request(quote_request)
    
    logger.info(f"Quote request created: {saved_request.id} from {form.company}")
    
    return {
        "success": True,
        "quote_reference": saved_request.id,
        "message": "Quote request received. Our sales team will send you a detailed quote within 24 hours.",
        "estimated_response": "24 hours",
        "contact_email": form.email,
        # Note: No pricing information exposed here
        "items_submitted": len(line_items),
        "status": saved_request.status.value
    }


@router.post(
    "/track-order",
    summary="Track order status",
    description="Look up order status by order number and email.",
    response_model=OrderTrackingResponse
)
async def track_order(request: OrderTrackingRequest) -> OrderTrackingResponse:
    """
    Track an order.
    
    Args:
        request: Order tracking request
        
    Returns:
        Order status information
    """
    # In production: query order database
    
    # Demo response
    if request.order_number.startswith("ORD-"):
        return {
            "success": True,
            "order_number": request.order_number,
            "status": "In Transit",
            "estimated_delivery": "2024-12-20",
            "tracking_number": "1Z999AA10123456784",
            "carrier": "UPS",
            "last_update": datetime.now().isoformat(),
            "history": [
                {"date": "2024-12-15", "status": "Order Placed"},
                {"date": "2024-12-16", "status": "Processing"},
                {"date": "2024-12-17", "status": "Shipped"},
                {"date": "2024-12-18", "status": "In Transit"}
            ]
        }
    
    return {
        "success": False,
        "message": "Order not found. Please check the order number and try again.",
        "order_number": request.order_number
    }


@router.get(
    "/faq",
    summary="Get FAQ answers",
    description="Get answer to frequently asked questions.",
    response_model=FAQResponse
)
async def get_faq_answer(
    question: str = Query(..., min_length=5, max_length=500)
) -> FAQResponse:
    """
    Get FAQ answer.
    
    Args:
        question: FAQ question
        
    Returns:
        Answer from knowledge base
    """
    result = await response_generator.generate_faq_answer(question)
    
    return {
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "source": result["source"]
    }


@router.get(
    "/session/{session_id}",
    summary="Get session info",
    description="Get information about a customer session.",
    response_model=SessionResponse
)
async def get_session(session_id: str) -> SessionResponse:
    """
    Get session information.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session data
    """
    if session_id not in customer_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = customer_sessions[session_id]
    current_node = decision_tree.get_node(session["current_node"])
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "current_node": current_node.to_dict() if current_node else None,
        "collected_data": session["collected_data"],
        "history_length": len(session.get("history", [])),
        "submissions": session.get("submissions", [])
    }
