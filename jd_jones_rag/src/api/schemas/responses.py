"""
API Response Schemas
Centralized Pydantic models for API responses to ensure proper OpenAPI documentation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Base Response Models
class BaseSuccessResponse(BaseModel):
    """Base model for simple success responses."""
    success: bool = True
    message: Optional[str] = None

# ==========================================
# Enquiry Responses
# ==========================================

class EnquirySubmitResponse(BaseSuccessResponse):
    enquiry_id: str
    estimated_response: str

class EnquiryStatsResponse(BaseModel):
    total: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    by_priority: Dict[str, int]
    recent_growth: float

class AIQuickView(BaseModel):
    one_liner: str
    type: str
    priority: str
    main_ask: str
    actions_needed: Dict[str, bool]
    urgency: Optional[int] = None
    products: Optional[List[str]] = None

class EnquiryQuickView(BaseModel):
    id: str
    customer: Dict[str, Any]
    subject: Optional[str]
    status: str
    created_at: str
    assigned_to: Optional[str]
    ai_quick_view: AIQuickView

class EnquiryDashboardResponse(BaseModel):
    enquiries: List[EnquiryQuickView]
    stats: Dict[str, Any]
    retrieved_at: str

class EnquiryAnalysisResponse(BaseSuccessResponse):
    enquiry_id: str
    analysis: Dict[str, Any]
    quick_view: Dict[str, Any]

class EnquiryDetailsResponse(BaseModel):
    enquiry: Dict[str, Any]
    viewed_by: str
    viewed_at: str

class AssignmentResponse(BaseSuccessResponse):
    enquiry_id: str

class NoteResponse(BaseSuccessResponse):
    notes_count: int

class GeneratedResponseModel(BaseModel):
    """Model for the AI generated response content."""
    subject: str
    body: str
    tone: str
    language: str

class GenerateResponseResponse(BaseSuccessResponse):
    suggested_response: Dict[str, Any]
    total_suggestions: int

class SendResponseResponse(BaseSuccessResponse):
    enquiry_id: str
    customer_email: str
    sent_at: str

class StatusUpdateResponse(BaseSuccessResponse):
    enquiry_id: str
    new_status: str

class EscalationResponse(BaseSuccessResponse):
    enquiry_id: str
    priority: str

class EnquiryDemoDashboardResponse(BaseModel):
    enquiries: List[Dict[str, Any]]
    stats: Dict[str, Any]
    retrieved_at: str

# ==========================================
# Quotation Responses
# ==========================================

class QuotationSubmitResponse(BaseSuccessResponse):
    request_id: str
    type: str
    status: str
    items_count: Optional[int] = None
    contact_email: Optional[str] = None
    estimated_response: Optional[str] = None
    quote_reference: Optional[str] = None
    items_submitted: Optional[int] = None
    ai_processing: Optional[bool] = None
    ai_summary: Optional[Dict[str, Any]] = None
    extracted_products: Optional[List[str]] = None

class QuotationStatsResponse(BaseModel):
    total: int
    by_status: Dict[str, int]
    recent: List[Dict[str, Any]]

class QuotationAnalysisResponse(BaseSuccessResponse):
    quotation_id: str
    analysis: Optional[Dict[str, Any]] = None
    quick_view: Optional[Dict[str, Any]] = None
    analyzed_by: Optional[str] = None
    analyzed_at: Optional[str] = None
    has_analysis: Optional[bool] = None

class QuotationDetailsResponse(BaseModel):
    quotation: Dict[str, Any]
    retrieved_by: str
    retrieved_at: str

class QuotationUpdateResponse(BaseSuccessResponse):
    quotation: Dict[str, Any]
    updated_by: str
    updated_at: str

class QuotationSentResponse(BaseSuccessResponse):
    quotation_number: Optional[str]
    status: str

class QuotationDashboardResponse(BaseModel):
    quotations: List[Dict[str, Any]]
    stats: Dict[str, Any]
    retrieved_at: str

class QuotationDemoDashboardResponse(BaseModel):
    quotations: List[Dict[str, Any]]
    stats: Dict[str, Any]
    retrieved_at: str

class QuotationListResponse(BaseModel):
    total: int
    quotations: List[Dict[str, Any]] # Or a specific QuotationSummaryModel if we want to be strict

class PDFGenerationResponse(BaseModel):
    doc_id: str
    doc_type: str
    title: str
    filename: str
    download_url: str
    format: str
    created_at: str

class MarkSentResponse(BaseSuccessResponse):
    quotation_id: str
    status: str

class SavePricesResponse(BaseSuccessResponse):
    quotation_id: str
    updated_items: int
    total_amount: float
    grand_total: float

# ==========================================
# External Portal Responses
# ==========================================

class DecisionTreeNodeResponse(BaseModel):
    node_id: str
    node_type: str
    content: str
    options: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class DecisionTreeResponse(BaseModel):
    root_node: Dict[str, Any]
    tree_structure: Dict[str, Any]

class NavigationResponse(BaseModel):
    session_id: str
    current_node: Dict[str, Any]
    collected_data: Dict[str, Any]
    history_length: int

class ClassificationResponse(BaseModel):
    session_id: str
    classification: Dict[str, Any]
    suggested_node: Dict[str, Any]
    intent_description: str

class QueryProcessResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    confidence: float
    sources: List[Dict[str, Any]]
    suggested_actions: List[str]
    suggested_node: Dict[str, Any]
    conversation_turn: int

class FormSubmissionResponse(BaseSuccessResponse):
    submission_id: str
    action_type: str
    next_steps: str

class ContactFormResponse(BaseSuccessResponse):
    submission_id: str
    contact_info: Dict[str, str]

class QuoteRequestResponse(BaseSuccessResponse):
    quote_reference: str
    estimated_response: str
    contact_email: str
    items_submitted: int
    status: str

class OrderTrackingResponse(BaseSuccessResponse):
    order_number: str
    status: Optional[str] = None
    estimated_delivery: Optional[str] = None
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None
    last_update: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class FAQResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    current_node: Optional[Dict[str, Any]]
    collected_data: Dict[str, Any]
    history_length: int
    submissions: List[Dict[str, Any]]

# ==========================================
# Internal Chat Responses
# ==========================================

class ClearSessionResponse(BaseSuccessResponse):
    session_id: str

class KnowledgeStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    collections: Dict[str, int]
    last_updated: Optional[str]
    user: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[Dict[str, Any]]

class UserProfileResponse(BaseModel):
    user: Dict[str, Any]
    permissions: List[str]
    accessible_departments: List[str]
