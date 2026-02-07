"""
Enquiry Models - Data models for customer enquiries and AI-assisted responses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class EnquiryType(str, Enum):
    """Type of customer enquiry."""
    PRODUCT_SELECTION = "product_selection"
    TECHNICAL_ASSISTANCE = "technical_assistance"
    PRICING = "pricing"
    ORDER_STATUS = "order_status"
    QUOTATION = "quotation"
    COMPLAINT = "complaint"
    GENERAL = "general"
    OTHER = "other"


class EnquiryPriority(str, Enum):
    """Priority level for enquiry."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class EnquiryStatus(str, Enum):
    """Status of an enquiry."""
    NEW = "new"
    UNDER_REVIEW = "under_review"
    AI_RESPONSE_GENERATED = "ai_response_generated"
    RESPONSE_SENT = "response_sent"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class CustomerDetails:
    """Customer information for enquiry."""
    name: str
    email: str
    company: Optional[str] = None
    phone: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "phone": self.phone
        }


@dataclass
class StructuredRequirements:
    """Structured extraction of customer requirements."""
    # Technical requirements
    industry: Optional[str] = None
    application: Optional[str] = None
    equipment_type: Optional[str] = None
    operating_temperature: Optional[str] = None  # e.g., "200-400°C"
    operating_pressure: Optional[str] = None  # e.g., "up to 150 bar"
    media_handled: Optional[str] = None  # e.g., "steam, chemicals"
    certifications_needed: List[str] = field(default_factory=list)
    
    # Size/quantity requirements
    dimensions: Optional[str] = None
    quantity: Optional[str] = None
    
    # Timeline requirements
    delivery_urgency: Optional[str] = None  # e.g., "2 weeks", "ASAP", "flexible"
    project_deadline: Optional[str] = None
    
    # Budget indicators
    budget_mentioned: bool = False
    budget_range: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "industry": self.industry,
            "application": self.application,
            "equipment_type": self.equipment_type,
            "operating_temperature": self.operating_temperature,
            "operating_pressure": self.operating_pressure,
            "media_handled": self.media_handled,
            "certifications_needed": self.certifications_needed,
            "dimensions": self.dimensions,
            "quantity": self.quantity,
            "delivery_urgency": self.delivery_urgency,
            "project_deadline": self.project_deadline,
            "budget_mentioned": self.budget_mentioned,
            "budget_range": self.budget_range
        }


@dataclass
class AIAnalysis:
    """
    AI-generated structured analysis of an enquiry.
    
    Provides a comprehensive breakdown with key fields so internal team
    can quickly understand the enquiry without reading the full message.
    """
    # Quick overview
    summary: str  # 2-3 sentence summary
    one_liner: str = ""  # Ultra-brief one-line summary
    
    # Classification
    detected_type: EnquiryType = EnquiryType.GENERAL
    detected_priority: EnquiryPriority = EnquiryPriority.MEDIUM
    confidence_score: float = 0.0  # 0-1 confidence in classification
    
    # Structured breakdown - KEY FIELDS FOR QUICK SCANNING
    customer_intent: str = ""  # What does the customer want?
    main_ask: str = ""  # The primary request/question
    secondary_asks: List[str] = field(default_factory=list)  # Additional requests
    
    # Extracted requirements
    requirements: Optional[StructuredRequirements] = None
    
    # Key information points
    key_points: List[str] = field(default_factory=list)
    
    # Product recommendations
    suggested_products: List[str] = field(default_factory=list)
    product_match_confidence: str = "low"  # low, medium, high
    
    # Sentiment & urgency
    sentiment: str = "neutral"  # positive, neutral, negative
    urgency_indicators: List[str] = field(default_factory=list)
    urgency_score: int = 1  # 1-5 scale
    
    # Action items for internal team
    recommended_actions: List[str] = field(default_factory=list)
    requires_technical_review: bool = False
    requires_pricing: bool = False
    requires_samples: bool = False
    
    # Sub-agent analysis results (for multi-agent approach)
    sub_agent_analyses: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            # Overview
            "summary": self.summary,
            "one_liner": self.one_liner,
            
            # Classification
            "detected_type": self.detected_type.value,
            "detected_priority": self.detected_priority.value,
            "confidence_score": self.confidence_score,
            
            # Structured breakdown
            "customer_intent": self.customer_intent,
            "main_ask": self.main_ask,
            "secondary_asks": self.secondary_asks,
            
            # Requirements
            "requirements": self.requirements.to_dict() if self.requirements else None,
            
            # Key info
            "key_points": self.key_points,
            
            # Products
            "suggested_products": self.suggested_products,
            "product_match_confidence": self.product_match_confidence,
            
            # Sentiment & urgency
            "sentiment": self.sentiment,
            "urgency_indicators": self.urgency_indicators,
            "urgency_score": self.urgency_score,
            
            # Actions
            "recommended_actions": self.recommended_actions,
            "requires_technical_review": self.requires_technical_review,
            "requires_pricing": self.requires_pricing,
            "requires_samples": self.requires_samples,
            
            # Sub-agent results
            "sub_agent_analyses": self.sub_agent_analyses
        }
    
    def get_quick_view(self) -> Dict[str, Any]:
        """Get a quick-scan view for internal team dashboard."""
        return {
            "one_liner": self.one_liner,
            "type": self.detected_type.value,
            "priority": self.detected_priority.value,
            "urgency": self.urgency_score,
            "main_ask": self.main_ask,
            "products": self.suggested_products[:3],
            "actions_needed": {
                "technical_review": self.requires_technical_review,
                "pricing": self.requires_pricing,
                "samples": self.requires_samples
            }
        }


@dataclass
class SuggestedResponse:
    """AI-generated suggested response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    response_text: str = ""
    tone: str = "professional"  # professional, friendly, formal
    generated_at: datetime = field(default_factory=datetime.now)
    approved: bool = False
    modified_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "response_text": self.response_text,
            "tone": self.tone,
            "generated_at": self.generated_at.isoformat(),
            "approved": self.approved,
            "modified_by": self.modified_by
        }


@dataclass
class Enquiry:
    """A customer enquiry with AI analysis and response tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Customer info
    customer: CustomerDetails = None
    
    # Raw enquiry
    raw_message: str = ""
    subject: Optional[str] = None
    session_id: Optional[str] = None
    
    # Classification
    enquiry_type: EnquiryType = EnquiryType.GENERAL
    priority: EnquiryPriority = EnquiryPriority.MEDIUM
    
    # AI Analysis
    ai_analysis: Optional[AIAnalysis] = None
    
    # Suggested responses
    suggested_responses: List[SuggestedResponse] = field(default_factory=list)
    
    # Response tracking
    final_response: Optional[str] = None
    response_sent_at: Optional[datetime] = None
    response_sent_by: Optional[str] = None
    
    # Status tracking
    status: EnquiryStatus = EnquiryStatus.NEW
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Assignment
    assigned_to: Optional[str] = None
    department: Optional[str] = None
    
    # Internal notes
    internal_notes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "customer": self.customer.to_dict() if self.customer else None,
            "subject": self.subject,
            "enquiry_type": self.enquiry_type.value,
            "priority": self.priority.value,
            "ai_analysis": self.ai_analysis.to_dict() if self.ai_analysis else None,
            "suggested_responses": [r.to_dict() for r in self.suggested_responses],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "department": self.department,
            "response_sent_at": self.response_sent_at.isoformat() if self.response_sent_at else None,
            "response_sent_by": self.response_sent_by
        }
        
        if include_raw:
            result["raw_message"] = self.raw_message
            result["final_response"] = self.final_response
            result["internal_notes"] = self.internal_notes
        
        return result
    
    def add_note(self, note: str, author: str):
        """Add an internal note."""
        self.internal_notes.append({
            "note": note,
            "author": author,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()


# In-memory storage (use database in production)
_enquiries: Dict[str, Enquiry] = {}


def save_enquiry(enquiry: Enquiry) -> Enquiry:
    """Save an enquiry."""
    enquiry.updated_at = datetime.now()
    _enquiries[enquiry.id] = enquiry
    return enquiry


def get_enquiry(enquiry_id: str) -> Optional[Enquiry]:
    """Get an enquiry by ID."""
    return _enquiries.get(enquiry_id)


def get_all_enquiries(
    status: Optional[EnquiryStatus] = None,
    enquiry_type: Optional[EnquiryType] = None,
    assigned_to: Optional[str] = None
) -> List[Enquiry]:
    """Get all enquiries with optional filters."""
    enquiries = list(_enquiries.values())
    
    if status:
        enquiries = [e for e in enquiries if e.status == status]
    if enquiry_type:
        enquiries = [e for e in enquiries if e.enquiry_type == enquiry_type]
    if assigned_to:
        enquiries = [e for e in enquiries if e.assigned_to == assigned_to]
    
    return sorted(enquiries, key=lambda x: x.created_at, reverse=True)


def get_enquiry_stats() -> Dict[str, Any]:
    """Get enquiry statistics."""
    all_enquiries = list(_enquiries.values())
    
    stats = {
        "total": len(all_enquiries),
        "by_status": {},
        "by_type": {},
        "by_priority": {}
    }
    
    for status in EnquiryStatus:
        stats["by_status"][status.value] = len([e for e in all_enquiries if e.status == status])
    
    for etype in EnquiryType:
        stats["by_type"][etype.value] = len([e for e in all_enquiries if e.enquiry_type == etype])
    
    for priority in EnquiryPriority:
        stats["by_priority"][priority.value] = len([e for e in all_enquiries if e.priority == priority])
    
    return stats
