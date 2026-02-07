"""
Quotation Models - Data models for quotation requests and generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class QuotationStatus(str, Enum):
    """Status of a quotation request."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    QUOTED = "quoted"
    SENT = "sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class QuotationLineItem:
    """
    A single product line in a quotation.
    
    For specific quotations: Customer fills these fields
    For generic quotations: AI agent suggests these fields (is_ai_suggested=True)
    """
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    product_code: str = ""  # e.g., "NA 710"
    product_name: str = ""
    
    # Size specifications
    size: Optional[str] = None  # e.g., "12mm × 12mm" or "1/2 inch"
    size_od: Optional[float] = None  # Outer diameter in mm
    size_id: Optional[float] = None  # Inner diameter in mm
    size_th: Optional[float] = None  # Thickness in mm
    dimension_unit: str = "mm"  # Unit for dimensions: "mm" or "inch"
    
    # Style and material
    style: Optional[str] = None  # e.g., "Braided", "Die-formed", "Twisted"
    material_grade: Optional[str] = None  # e.g., "Standard", "High Purity", "Food Grade"
    material_code: Optional[str] = None  # Internal material code
    
    # Dimensions table (for complex products)
    dimensions: Optional[Dict[str, Any]] = None  # {"length": "500mm", "width": "6mm", "custom": "..."}
    
    # Quantity
    quantity: int = 1
    unit: str = "Nos."
    rings_per_set: Optional[int] = None
    
    # Specific requirements (free text)
    specific_requirements: Optional[str] = None  # e.g., "Need in black colour, fire-safe certification"
    colour: Optional[str] = None  # e.g., "Black", "Natural/Grey", "Custom"
    
    # Pricing (internal use only)
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    
    # AI suggestion flag
    is_ai_suggested: bool = False  # True if AI filled this, False if customer specified
    ai_confidence: Optional[float] = None  # 0.0 to 1.0 confidence score for AI suggestions
    
    # Notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without pricing for external use)."""
        return {
            "id": self.id,
            "product_code": self.product_code,
            "product_name": self.product_name,
            "size": self.size,
            "size_od": self.size_od,
            "size_id": self.size_id,
            "size_th": self.size_th,
            "dimension_unit": self.dimension_unit,
            "style": self.style,
            "material_grade": self.material_grade,
            "material_code": self.material_code,
            "dimensions": self.dimensions,
            "quantity": self.quantity,
            "unit": self.unit,
            "rings_per_set": self.rings_per_set,
            "specific_requirements": self.specific_requirements,
            "colour": self.colour,
            "is_ai_suggested": self.is_ai_suggested,
            "ai_confidence": self.ai_confidence,
            "notes": self.notes
        }
    
    def to_internal_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with pricing (internal use only)."""
        result = self.to_dict()
        result["unit_price"] = self.unit_price
        result["amount"] = self.amount
        return result


@dataclass
class CustomerInfo:
    """Customer information for quotation."""
    name: str
    company: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    designation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "designation": self.designation
        }


@dataclass
class QuotationRequest:
    """A customer quotation request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    quotation_number: Optional[str] = None  # Generated when quoted
    customer: CustomerInfo = None
    reference_rfq: Optional[str] = None  # Customer's RFQ number
    rfq_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Requirements
    industry: Optional[str] = None
    application: Optional[str] = None
    operating_conditions: Optional[str] = None
    special_requirements: Optional[str] = None
    
    # Line items
    line_items: List[QuotationLineItem] = field(default_factory=list)
    
    # Status tracking
    status: QuotationStatus = QuotationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    quoted_at: Optional[datetime] = None
    
    # Internal fields (never exposed externally)
    total_amount: Optional[float] = None
    gst_amount: Optional[float] = None
    grand_total: Optional[float] = None
    internal_notes: Optional[str] = None
    assigned_to: Optional[str] = None  # Sales rep
    
    # AI Processing flag
    # True = Customer submitted generic request, needs AI to extract requirements
    # False = Customer provided specific requirements, ready for direct processing
    requires_ai_processing: bool = False
    ai_processed: bool = False  # Has AI already processed this?
    original_message: Optional[str] = None  # Customer's original free-text message
    
    def to_dict(self, include_internal: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "quotation_number": self.quotation_number,
            "customer": self.customer.to_dict() if self.customer else None,
            "reference_rfq": self.reference_rfq,
            "rfq_date": self.rfq_date.isoformat() if self.rfq_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "industry": self.industry,
            "application": self.application,
            "operating_conditions": self.operating_conditions,
            "special_requirements": self.special_requirements,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "quoted_at": self.quoted_at.isoformat() if self.quoted_at else None
        }
        
        if include_internal:
            result["line_items"] = [item.to_internal_dict() for item in self.line_items]
            result["total_amount"] = self.total_amount
            result["gst_amount"] = self.gst_amount
            result["grand_total"] = self.grand_total
            result["internal_notes"] = self.internal_notes
            result["assigned_to"] = self.assigned_to
        else:
            result["line_items"] = [item.to_dict() for item in self.line_items]
            result["item_count"] = len(self.line_items)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuotationRequest":
        """Create from dictionary."""
        customer_data = data.get("customer", {})
        customer = CustomerInfo(**customer_data) if customer_data else None
        
        line_items = []
        for item_data in data.get("line_items", []):
            line_items.append(QuotationLineItem(**item_data))
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            quotation_number=data.get("quotation_number"),
            customer=customer,
            reference_rfq=data.get("reference_rfq"),
            rfq_date=datetime.fromisoformat(data["rfq_date"]) if data.get("rfq_date") else None,
            due_date=datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None,
            industry=data.get("industry"),
            application=data.get("application"),
            operating_conditions=data.get("operating_conditions"),
            special_requirements=data.get("special_requirements"),
            line_items=line_items,
            status=QuotationStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


# In-memory storage (use database in production)
_quotation_requests: Dict[str, QuotationRequest] = {}
_quotation_counter: int = 65471  # Start from sample quotation number


def save_quotation_request(request: QuotationRequest) -> QuotationRequest:
    """Save a quotation request."""
    request.updated_at = datetime.now()
    _quotation_requests[request.id] = request
    print(f"[STORAGE] Saved quotation {request.id} - Total quotations now: {len(_quotation_requests)}", flush=True)
    return request


def get_quotation_request(request_id: str) -> Optional[QuotationRequest]:
    """Get a quotation request by ID."""
    return _quotation_requests.get(request_id)


def get_all_quotation_requests(status: Optional[QuotationStatus] = None) -> List[QuotationRequest]:
    """Get all quotation requests, optionally filtered by status."""
    print(f"[STORAGE] get_all_quotation_requests called - {len(_quotation_requests)} in storage", flush=True)
    requests = list(_quotation_requests.values())
    if status:
        requests = [r for r in requests if r.status == status]
    return sorted(requests, key=lambda x: x.created_at, reverse=True)


def generate_quotation_number() -> str:
    """Generate a unique quotation number."""
    global _quotation_counter
    _quotation_counter += 1
    return str(_quotation_counter)
