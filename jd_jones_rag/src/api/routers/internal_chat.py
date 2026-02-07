"""
Internal Chat Router - Employee Chatbot API Endpoints
Handles authenticated employee interactions with the RAG system.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from src.auth.authentication import User, get_current_active_user
from src.auth.authorization import Permission, require_permission, AuthorizationService
from src.agents.internal_agent import InternalAgent


router = APIRouter()

# Initialize agent
internal_agent = InternalAgent()


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the safety guidelines for operating the CNC machine?",
                "session_id": "session_abc123"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Dict[str, Any]


class SessionHistoryResponse(BaseModel):
    """Session history response."""
    session_id: str
    messages: List[Dict[str, str]]
    created_at: Optional[str]


class SuggestedQuestionsResponse(BaseModel):
    """Suggested questions response."""
    questions: List[str]
    user_role: str


# Endpoints
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="Send a message to the internal AI assistant and receive a response with sources."
)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(require_permission(Permission.CHAT_WITH_AGENT))
) -> ChatResponse:
    """
    Process a chat message from an authenticated employee.
    
    Args:
        request: Chat request with message and optional session ID
        current_user: Authenticated user
        
    Returns:
        AI response with sources and metadata
    """
    try:
        result = await internal_agent.chat(
            user=current_user,
            query=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            session_id=result["session_id"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get conversation history",
    description="Retrieve the conversation history for a specific session."
)
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
) -> SessionHistoryResponse:
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        current_user: Authenticated user
        
    Returns:
        Session history with messages
    """
    history = internal_agent.get_session_history(session_id)
    
    return SessionHistoryResponse(
        session_id=session_id,
        messages=history,
        created_at=datetime.now().isoformat() if history else None
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Clear session",
    description="Clear the conversation history for a session."
)
async def clear_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Clear a conversation session.
    
    Args:
        session_id: Session identifier
        current_user: Authenticated user
        
    Returns:
        Confirmation message
    """
    cleared = internal_agent.clear_session(session_id)
    
    return {
        "success": cleared,
        "session_id": session_id,
        "message": "Session cleared" if cleared else "Session not found"
    }


@router.get(
    "/suggestions",
    response_model=SuggestedQuestionsResponse,
    summary="Get suggested questions",
    description="Get personalized suggested questions based on user role."
)
async def get_suggestions(
    current_user: User = Depends(get_current_active_user)
) -> SuggestedQuestionsResponse:
    """
    Get suggested questions for the user.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        List of suggested questions
    """
    suggestions = await internal_agent.get_suggested_questions(current_user)
    
    return SuggestedQuestionsResponse(
        questions=suggestions,
        user_role=current_user.role
    )


@router.get(
    "/knowledge-stats",
    summary="Get knowledge base statistics",
    description="Get statistics about accessible knowledge bases."
)
async def get_knowledge_stats(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get knowledge base statistics for the user.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Statistics about accessible knowledge bases
    """
    from src.knowledge_base.retriever import HierarchicalRetriever, UserRole
    
    retriever = HierarchicalRetriever()
    
    try:
        user_role = UserRole(current_user.role)
    except ValueError:
        user_role = UserRole.EMPLOYEE
    
    stats = retriever.get_retrieval_stats(user_role)
    stats["user"] = {
        "name": current_user.full_name,
        "role": current_user.role,
        "department": current_user.department
    }
    
    return stats


@router.get(
    "/search",
    summary="Search knowledge base",
    description="Search the knowledge base without conversational context."
)
async def search_knowledge_base(
    query: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(require_permission(Permission.SEARCH_DOCUMENTS))
) -> Dict[str, Any]:
    """
    Search the knowledge base directly.
    
    Args:
        query: Search query
        limit: Maximum results
        current_user: Authenticated user
        
    Returns:
        Search results
    """
    from src.knowledge_base.retriever import HierarchicalRetriever, UserRole
    
    retriever = HierarchicalRetriever()
    
    try:
        user_role = UserRole(current_user.role)
    except ValueError:
        user_role = UserRole.EMPLOYEE
    
    results = retriever.retrieve(
        query=query,
        user_role=user_role,
        user_department=current_user.department,
        n_results=limit
    )
    
    return {
        "query": query,
        "total_results": results.total_count,
        "results": [r.to_dict() for r in results.all_results]
    }


@router.get(
    "/user/profile",
    summary="Get user profile",
    description="Get the current user's profile and permissions."
)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get current user profile and permissions.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        User profile with permissions
    """
    permissions = AuthorizationService.get_user_permissions(current_user)
    accessible_depts = AuthorizationService.get_accessible_departments(current_user)
    
    return {
        "user": current_user.to_dict(),
        "permissions": [p.value for p in permissions],
        "accessible_departments": [d.value for d in accessible_depts]
    }
