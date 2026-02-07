"""
Admin Router - Administration API Endpoints
Handles document management, user management, and system administration.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from src.auth.authentication import User, get_current_active_user, create_user
from src.auth.authorization import Permission, require_permission
from src.config.settings import settings
from src.data_ingestion.document_processor import DocumentProcessor, AccessLevel
from src.data_ingestion.embedding_generator import EmbeddingGenerator
from src.data_ingestion.vector_store import VectorStoreManager
from src.knowledge_base.retriever import HierarchicalRetriever


router = APIRouter()

# Initialize components
document_processor = DocumentProcessor()
embedding_generator = EmbeddingGenerator()
vector_store = VectorStoreManager()


# Request/Response Models
class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    success: bool
    document_id: str
    file_name: str
    chunks_created: int
    collection: str


class UserCreateRequest(BaseModel):
    """User creation request."""
    email: str
    password: str
    full_name: str
    role: str
    department: Optional[str] = None
    is_internal: bool = True


class CollectionStatsResponse(BaseModel):
    """Collection statistics response."""
    collection_name: str
    document_count: int
    status: str


# Document Management Endpoints
@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload document",
    description="Upload and process a document into the knowledge base."
)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(...),
    access_level: str = Form(default="level_0_internal"),
    department: Optional[str] = Form(default=None),
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    Args:
        file: Document file
        collection: Target collection name
        access_level: Access level for the document
        department: Department if applicable
        current_user: Authenticated admin user
        
    Returns:
        Upload result with document info
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed: {settings.allowed_extensions}"
        )
    
    # Save file temporarily
    temp_path = settings.upload_path / f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse access level
        try:
            doc_access_level = AccessLevel(access_level)
        except ValueError:
            doc_access_level = AccessLevel.LEVEL_0_INTERNAL
        
        # Process document
        processed_docs = document_processor.process_file(
            str(temp_path),
            access_level=doc_access_level,
            department=department,
            additional_metadata={
                "uploaded_by": current_user.user_id,
                "upload_date": datetime.now().isoformat()
            }
        )
        
        # Generate embeddings
        embedded_docs = embedding_generator.process_documents(
            processed_docs,
            show_progress=False
        )
        
        # Store in vector store
        chunks_added = vector_store.add_documents(collection, embedded_docs)
        
        return DocumentUploadResponse(
            success=True,
            document_id=processed_docs[0].document_id if processed_docs else "",
            file_name=file.filename,
            chunks_created=chunks_added,
            collection=collection
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            os.remove(temp_path)


@router.delete(
    "/documents/{collection}/{document_id}",
    summary="Delete document",
    description="Delete a document from the knowledge base."
)
async def delete_document(
    collection: str,
    document_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Delete a document from a collection.
    
    Args:
        collection: Collection name
        document_id: Document ID
        current_user: Authenticated admin user
        
    Returns:
        Deletion confirmation
    """
    success = vector_store.delete_documents(
        collection,
        where={"document_id": document_id}
    )
    
    return {
        "success": success,
        "document_id": document_id,
        "collection": collection,
        "deleted_by": current_user.user_id
    }


@router.get(
    "/documents/collections",
    summary="List collections",
    description="List all document collections."
)
async def list_collections(
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> Dict[str, Any]:
    """
    List all collections in the vector store.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        List of collections with stats
    """
    collections = vector_store.list_collections()
    
    collection_stats = []
    for name in collections:
        count = vector_store.get_collection_count(name)
        collection_stats.append({
            "name": name,
            "document_count": count,
            "status": "active"
        })
    
    return {
        "total_collections": len(collections),
        "collections": collection_stats
    }


@router.get(
    "/documents/collections/{collection}/stats",
    response_model=CollectionStatsResponse,
    summary="Get collection statistics",
    description="Get detailed statistics for a collection."
)
async def get_collection_stats(
    collection: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_DOCUMENTS))
) -> CollectionStatsResponse:
    """
    Get statistics for a specific collection.
    
    Args:
        collection: Collection name
        current_user: Authenticated admin user
        
    Returns:
        Collection statistics
    """
    count = vector_store.get_collection_count(collection)
    
    return CollectionStatsResponse(
        collection_name=collection,
        document_count=count,
        status="active" if count > 0 else "empty"
    )


@router.post(
    "/documents/reindex/{collection}",
    summary="Reindex collection",
    description="Reindex all documents in a collection."
)
async def reindex_collection(
    collection: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_ACCESS))
) -> Dict[str, Any]:
    """
    Trigger reindexing of a collection.
    
    Args:
        collection: Collection name
        current_user: Authenticated admin user
        
    Returns:
        Reindex status
    """
    # In production, this would trigger a background task
    return {
        "status": "initiated",
        "collection": collection,
        "message": "Reindexing has been queued. This may take several minutes.",
        "initiated_by": current_user.user_id
    }


# User Management Endpoints
@router.post(
    "/users",
    summary="Create user",
    description="Create a new user account."
)
async def create_new_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
) -> Dict[str, Any]:
    """
    Create a new user.
    
    Args:
        request: User creation data
        current_user: Authenticated admin user
        
    Returns:
        Created user info
    """
    try:
        user = create_user(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            role=request.role,
            department=request.department,
            is_internal=request.is_internal
        )
        
        return {
            "success": True,
            "user": user.to_dict(),
            "created_by": current_user.user_id
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/users",
    summary="List users",
    description="List all users in the system."
)
async def list_users(
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
) -> Dict[str, Any]:
    """
    List all users.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        List of users
    """
    from src.auth.authentication import USERS_DB
    
    users = [user.to_dict() for user in USERS_DB.values()]
    
    return {
        "total_users": len(users),
        "users": users
    }


# System Administration Endpoints
@router.get(
    "/system/stats",
    summary="Get system statistics",
    description="Get overall system statistics."
)
async def get_system_stats(
    current_user: User = Depends(require_permission(Permission.ADMIN_ACCESS))
) -> Dict[str, Any]:
    """
    Get system-wide statistics.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        System statistics
    """
    from src.auth.authentication import USERS_DB
    
    collections = vector_store.list_collections()
    total_docs = sum(
        vector_store.get_collection_count(c) for c in collections
    )
    
    return {
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        },
        "knowledge_base": {
            "total_collections": len(collections),
            "total_documents": total_docs,
            "collections": collections
        },
        "users": {
            "total_users": len(USERS_DB),
            "internal_users": sum(1 for u in USERS_DB.values() if u.is_internal),
            "external_users": sum(1 for u in USERS_DB.values() if not u.is_internal)
        },
        "generated_at": datetime.now().isoformat()
    }


@router.get(
    "/system/config",
    summary="Get system configuration",
    description="Get current system configuration (non-sensitive)."
)
async def get_system_config(
    current_user: User = Depends(require_permission(Permission.ADMIN_ACCESS))
) -> Dict[str, Any]:
    """
    Get system configuration.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        Non-sensitive configuration values
    """
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_retrieval_results": settings.max_retrieval_results,
        "jwt_expiration_minutes": settings.jwt_access_token_expire_minutes,
        "allowed_file_extensions": settings.allowed_extensions_list
    }


@router.post(
    "/system/clear-cache",
    summary="Clear system cache",
    description="Clear all system caches."
)
async def clear_cache(
    current_user: User = Depends(require_permission(Permission.ADMIN_ACCESS))
) -> Dict[str, Any]:
    """
    Clear system caches.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        Cache clear status
    """
    # In production: clear Redis cache, session cache, etc.
    
    return {
        "success": True,
        "message": "Cache cleared successfully",
        "cleared_by": current_user.user_id,
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/knowledge-base/overview",
    summary="Get knowledge base overview",
    description="Get overview of all knowledge bases and their statistics."
)
async def get_knowledge_base_overview(
    current_user: User = Depends(require_permission(Permission.ADMIN_ACCESS))
) -> Dict[str, Any]:
    """
    Get complete knowledge base overview.
    
    Args:
        current_user: Authenticated admin user
        
    Returns:
        Knowledge base statistics
    """
    from src.knowledge_base.main_context import MainContextDatabase
    from src.knowledge_base.level_contexts import LevelContextDatabase, Department
    
    main_db = MainContextDatabase()
    level_db = LevelContextDatabase()
    
    return {
        "main_context": {
            "internal_documents": main_db.get_document_count(False),
            "public_documents": main_db.get_document_count(True)
        },
        "department_contexts": level_db.get_all_document_counts(),
        "total_documents": (
            main_db.get_document_count(False) +
            sum(level_db.get_all_document_counts().values())
        ),
        "generated_at": datetime.now().isoformat()
    }


# In-memory enquiries store (in production, use database)
_enquiries_store: List[Dict[str, Any]] = []


@router.get(
    "/enquiries",
    summary="List enquiries",
    description="List all customer enquiries."
)
async def list_enquiries(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, le=200),
) -> Dict[str, Any]:
    """
    List customer enquiries.
    
    This endpoint is public for internal portal access.
    In production, implement proper authentication.
    
    Returns:
        List of enquiries
    """
    global _enquiries_store
    
    # Return stored enquiries or demo data
    if _enquiries_store:
        enquiries = _enquiries_store
    else:
        # Return demo data for initial setup
        enquiries = [
            {
                "enquiry_id": "ENQ-20260204-001",
                "from_email": "customer@example.com",
                "from_name": "John Smith",
                "company": "ABC Industries",
                "content": "Need quote for NA 701 graphite packing, 50 kg quantity for steam valve application.",
                "category": "quote_request",
                "status": "pending",
                "priority": "high",
                "created_at": datetime.now().isoformat(),
                "routed_to": "Sales Team"
            },
            {
                "enquiry_id": "ENQ-20260204-002",
                "from_email": "engineer@petrochemical.com",
                "from_name": "Sarah Johnson",
                "company": "PetroChem Ltd",
                "content": "What certifications does NA 750 have for fugitive emissions?",
                "category": "technical",
                "status": "responded",
                "priority": "medium",
                "created_at": datetime.now().isoformat(),
                "response": "NA 750 is certified to API 622 and ISO 15848-1, making it suitable for fugitive emission control applications.",
                "routed_to": "Technical Support"
            },
            {
                "enquiry_id": "ENQ-20260203-003",
                "from_email": "procurement@manufacturing.com",
                "from_name": "Mike Chen",
                "company": "Global Manufacturing",
                "content": "Looking for fire-safe valve packing for refinery application. Temperature up to 400°C.",
                "category": "product_info",
                "status": "pending",
                "priority": "high",
                "created_at": datetime.now().isoformat(),
                "routed_to": "Sales Team"
            }
        ]
    
    # Apply status filter if provided
    if status_filter and status_filter != "all":
        enquiries = [e for e in enquiries if e.get("status") == status_filter]
    
    return {
        "enquiries": enquiries[:limit],
        "total": len(enquiries)
    }


@router.post(
    "/enquiries",
    summary="Add enquiry",
    description="Add a new enquiry to the store (called after external enquiry submission)."
)
async def add_enquiry(
    enquiry_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add a new enquiry to the store.
    
    Called by the agentic system after processing an enquiry or from the product wizard.
    """
    global _enquiries_store
    
    enquiry_id = enquiry_data.get("enquiry_id", f"ENQ-{datetime.now().strftime('%Y%m%d')}-{len(_enquiries_store)+1:03d}")
    
    # Map frontend field names to expected names
    from_name = enquiry_data.get("from_name") or enquiry_data.get("customer_name", "Website Visitor")
    from_email = enquiry_data.get("from_email") or enquiry_data.get("customer_email", "")
    company = enquiry_data.get("company", "")
    
    # Build content from message and product details
    message = enquiry_data.get("content") or enquiry_data.get("message", "")
    product_code = enquiry_data.get("product_code", "")
    product_name = enquiry_data.get("product_name", "")
    source = enquiry_data.get("source", "")
    
    # Create rich content including product details
    content_parts = []
    if product_code or product_name:
        content_parts.append(f"**Product:** {product_name} ({product_code})")
    if message:
        content_parts.append(message)
    if enquiry_data.get("specifications"):
        specs = enquiry_data.get("specifications", {})
        specs_str = ", ".join([f"{k}: {v}" for k, v in specs.items() if v])
        if specs_str:
            content_parts.append(f"**Specifications:** {specs_str}")
    if enquiry_data.get("certifications"):
        content_parts.append(f"**Certifications:** {', '.join(enquiry_data.get('certifications', []))}")
    
    content = "\n".join(content_parts) if content_parts else "Quote request"
    
    # Determine category based on source
    category = enquiry_data.get("category", "quote_request" if source == "product_wizard" else "general")
    
    enquiry = {
        "enquiry_id": enquiry_id,
        "from_email": from_email,
        "from_name": from_name,
        "company": company,
        "content": content,
        "category": category,
        "status": enquiry_data.get("status", "pending"),
        "priority": enquiry_data.get("priority", "medium"),
        "created_at": datetime.now().isoformat(),
        "response": enquiry_data.get("response"),
        "routed_to": enquiry_data.get("routed_to", "Sales Team" if source == "product_wizard" else None),
        "product_code": product_code,
        "product_name": product_name,
        "source": source
    }
    
    _enquiries_store.insert(0, enquiry)  # Insert at beginning (newest first)
    
    return {
        "success": True,
        "enquiry_id": enquiry_id
    }

