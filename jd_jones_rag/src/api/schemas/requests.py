"""
API Request and Response Schemas
Pydantic models for API validation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, EmailStr, field_validator


# ==========================================
# Authentication Schemas
# ==========================================
class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


# ==========================================
# Chat Schemas
# ==========================================
class ChatMessageRole(str, Enum):
    """Chat message role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message schema."""
    role: ChatMessageRole
    content: str
    timestamp: Optional[datetime] = None


class InternalChatRequest(BaseModel):
    """Internal chat request."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the safety procedures for equipment X?",
                "session_id": "session_123"
            }
        }


class InternalChatResponse(BaseModel):
    """Internal chat response."""
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Dict[str, Any]


# ==========================================
# External Portal Schemas
# ==========================================
class ExternalQueryRequest(BaseModel):
    """External customer query request."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class NavigationOption(BaseModel):
    """Decision tree navigation option."""
    label: str
    next_node_id: str


class TreeNodeResponse(BaseModel):
    """Decision tree node response."""
    node_id: str
    node_type: str
    title: str
    content: str
    options: List[NavigationOption]
    form_fields: Optional[List[Dict[str, Any]]] = None


class FormSubmission(BaseModel):
    """Form submission schema."""
    node_id: str
    form_data: Dict[str, Any]
    session_id: Optional[str] = None


# ==========================================
# Document Schemas
# ==========================================
class DocumentMetadata(BaseModel):
    """Document metadata schema."""
    file_name: str
    file_type: str
    access_level: str
    department: Optional[str] = None
    uploaded_by: Optional[str] = None
    created_at: Optional[datetime] = None


class DocumentUploadRequest(BaseModel):
    """Document upload metadata."""
    collection: str = Field(..., min_length=1)
    access_level: str = "level_0_internal"
    department: Optional[str] = None
    tags: Optional[List[str]] = None


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., min_length=1, max_length=500)
    collections: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result schema."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str


class SearchResponse(BaseModel):
    """Search response schema."""
    query: str
    total_results: int
    results: List[SearchResult]


# ==========================================
# User Schemas
# ==========================================
class UserCreate(BaseModel):
    """User creation schema."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=100)
    role: str
    department: Optional[str] = None
    is_internal: bool = True
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UserResponse(BaseModel):
    """User response schema (no password)."""
    user_id: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    is_active: bool
    is_internal: bool


class UserUpdate(BaseModel):
    """User update schema."""
    full_name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None


# ==========================================
# Memory Schemas (Super Memory)
# ==========================================
class MemoryType(str, Enum):
    """Memory type enumeration."""
    FACT = "fact"
    PREFERENCE = "preference"
    CONTEXT = "context"
    INSTRUCTION = "instruction"
    ENTITY = "entity"


class Memory(BaseModel):
    """Memory schema."""
    content: str = Field(..., min_length=1, max_length=5000)
    memory_type: MemoryType
    category: Optional[str] = None
    importance_score: float = Field(default=0.5, ge=0, le=1)
    confidence_score: float = Field(default=0.8, ge=0, le=1)
    tags: Optional[List[str]] = None


class MemoryResponse(BaseModel):
    """Memory response schema."""
    memory_id: str
    content: str
    memory_type: str
    category: Optional[str]
    importance_score: float
    confidence_score: float
    source_provider: Optional[str]
    created_at: datetime
    accessed_at: Optional[datetime]


class MemorySyncRequest(BaseModel):
    """Memory sync request schema."""
    providers: List[str]
    full_sync: bool = False


class MemorySyncResponse(BaseModel):
    """Memory sync response schema."""
    sync_id: str
    status: str
    providers: Dict[str, Any]
    total_memories_added: int
    total_memories_updated: int
    total_memories_skipped: int


# ==========================================
# Quote Schemas
# ==========================================
class QuoteRequest(BaseModel):
    """Quote request schema."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=20)
    company: str = Field(..., min_length=1, max_length=200)
    products: str = Field(..., min_length=10, max_length=2000)
    quantity: Optional[int] = None
    delivery_date: Optional[str] = None
    budget_range: Optional[str] = None
    notes: Optional[str] = None


class QuoteResponse(BaseModel):
    """Quote response schema."""
    quote_reference: str
    status: str
    message: str
    estimated_response: str
    created_at: datetime


# ==========================================
# Support Ticket Schemas
# ==========================================
class TicketPriority(str, Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportTicketRequest(BaseModel):
    """Support ticket request schema."""
    name: str
    email: EmailStr
    phone: Optional[str] = None
    order_number: Optional[str] = None
    product: str
    issue_description: str = Field(..., min_length=20, max_length=5000)
    priority: TicketPriority = TicketPriority.MEDIUM


class SupportTicketResponse(BaseModel):
    """Support ticket response schema."""
    ticket_id: str
    status: str
    priority: str
    estimated_response_time: str
    created_at: datetime


# ==========================================
# Health Check Schemas
# ==========================================
class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    app: str
    version: str
    environment: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response."""
    services: Dict[str, str]
