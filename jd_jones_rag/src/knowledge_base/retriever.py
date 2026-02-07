"""
Hierarchical Retriever with Access Control.
Combines main context and level contexts with role-based access filtering.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from src.knowledge_base.main_context import MainContextDatabase, MainContextResult
from src.knowledge_base.level_contexts import (
    LevelContextDatabase, 
    Department, 
    DepartmentContextResult
)
from src.config.settings import settings


class UserRole(str, Enum):
    """User role identifiers."""
    EMPLOYEE = "employee"  # Basic internal access
    SALES_REP = "sales_rep"
    SALES_MANAGER = "sales_manager"
    PRODUCTION_WORKER = "production_worker"
    PRODUCTION_SUPERVISOR = "production_supervisor"
    ENGINEER = "engineer"
    ENGINEERING_MANAGER = "engineering_manager"
    CUSTOMER_SERVICE_REP = "customer_service_rep"
    CUSTOMER_SERVICE_MANAGER = "customer_service_manager"
    MANAGER = "manager"  # Cross-department access
    EXECUTIVE = "executive"  # Full access
    EXTERNAL_CUSTOMER = "external_customer"  # Public only


@dataclass
class RetrievalResult:
    """Unified retrieval result from any context."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    document_id: str
    chunk_index: int
    source: str  # 'main' or department name
    access_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "access_level": self.access_level
        }


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""
    query: str
    main_results: List[RetrievalResult] = field(default_factory=list)
    department_results: Dict[str, List[RetrievalResult]] = field(default_factory=dict)
    all_results: List[RetrievalResult] = field(default_factory=list)
    total_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "main_results": [r.to_dict() for r in self.main_results],
            "department_results": {
                k: [r.to_dict() for r in v] 
                for k, v in self.department_results.items()
            },
            "all_results": [r.to_dict() for r in self.all_results],
            "total_count": self.total_count
        }


class HierarchicalRetriever:
    """
    Hierarchical Retrieval System with Access Control.
    
    Retrieves from both main context (Level 0) and department contexts (Level 1+)
    based on user role and permissions.
    """
    
    # Role to department mapping
    ROLE_DEPARTMENT_ACCESS = {
        UserRole.EMPLOYEE: [],  # Main context only
        UserRole.SALES_REP: [Department.SALES],
        UserRole.SALES_MANAGER: [Department.SALES],
        UserRole.PRODUCTION_WORKER: [Department.PRODUCTION],
        UserRole.PRODUCTION_SUPERVISOR: [Department.PRODUCTION],
        UserRole.ENGINEER: [Department.ENGINEERING],
        UserRole.ENGINEERING_MANAGER: [Department.ENGINEERING],
        UserRole.CUSTOMER_SERVICE_REP: [Department.CUSTOMER_SERVICE],
        UserRole.CUSTOMER_SERVICE_MANAGER: [Department.CUSTOMER_SERVICE],
        UserRole.MANAGER: [
            Department.SALES, 
            Department.PRODUCTION, 
            Department.ENGINEERING,
            Department.CUSTOMER_SERVICE,
            Department.MANAGEMENT
        ],
        UserRole.EXECUTIVE: [
            Department.SALES, 
            Department.PRODUCTION, 
            Department.ENGINEERING,
            Department.CUSTOMER_SERVICE,
            Department.MANAGEMENT
        ],
        UserRole.EXTERNAL_CUSTOMER: [],  # Public only
    }
    
    def __init__(self):
        """Initialize hierarchical retriever with hybrid search."""
        self.main_context = MainContextDatabase()
        # Lazy import to avoid circular dependency
        from src.retrieval.enhanced_retrieval import HybridRetrieval
        self.hybrid_retrieval = HybridRetrieval(
            vector_weight=0.4,
            keyword_weight=0.6,
            exact_match_boost=2.0
        )
        self.level_contexts = LevelContextDatabase()
    
    def get_accessible_departments(
        self, 
        user_role: UserRole,
        user_department: Optional[str] = None
    ) -> List[Department]:
        """
        Get list of departments a user can access.
        
        Args:
            user_role: User's role
            user_department: User's primary department
            
        Returns:
            List of accessible departments
        """
        # Get departments from role mapping
        departments = list(self.ROLE_DEPARTMENT_ACCESS.get(user_role, []))
        
        # If user has a primary department, ensure it's included
        if user_department:
            try:
                user_dept = Department(user_department)
                if user_dept not in departments:
                    departments.append(user_dept)
            except ValueError:
                pass
        
        return departments
    
    def is_public_only_user(self, user_role: UserRole) -> bool:
        """Check if user should only see public content."""
        return user_role == UserRole.EXTERNAL_CUSTOMER
    
    def retrieve(
        self,
        query: str,
        user_role: UserRole,
        user_department: Optional[str] = None,
        n_results: int = 10,
        include_public_only: bool = False
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents based on user access.
        
        Args:
            query: Search query
            user_role: User's role for access control
            user_department: User's primary department
            n_results: Maximum total results
            include_public_only: Force public-only access
            
        Returns:
            RetrievalResponse with results from all accessible sources
        """
        response = RetrievalResponse(query=query)
        
        # Determine access level
        public_only = include_public_only or self.is_public_only_user(user_role)
        
        # Step 1: Query main context (Level 0) using hybrid search
        main_results_count = n_results if public_only else max(n_results // 2, 10)
        
        # Use hybrid retrieval for better keyword matching
        main_raw = self.hybrid_retrieval.hybrid_search(
            query=query,
            n_results=main_results_count,
            include_public_only=public_only
        )
        
        # Convert to unified format
        for mr in main_raw:
            result = RetrievalResult(
                content=mr.content,
                metadata=mr.metadata,
                relevance_score=mr.relevance_score,
                document_id=mr.document_id,
                chunk_index=mr.chunk_index,
                source="main",
                access_level="public" if mr.is_public else "internal"
            )
            response.main_results.append(result)
            response.all_results.append(result)
        
        # Step 2: Query department contexts (Level 1+) if not public-only
        if not public_only:
            accessible_depts = self.get_accessible_departments(user_role, user_department)
            
            if accessible_depts:
                dept_results_count = n_results - len(response.main_results)
                results_per_dept = max(2, dept_results_count // len(accessible_depts))
                
                for dept in accessible_depts:
                    dept_raw = self.level_contexts.query_department(
                        department=dept,
                        query_text=query,
                        n_results=results_per_dept
                    )
                    
                    dept_results = []
                    for dr in dept_raw:
                        result = RetrievalResult(
                            content=dr.content,
                            metadata=dr.metadata,
                            relevance_score=dr.relevance_score,
                            document_id=dr.document_id,
                            chunk_index=dr.chunk_index,
                            source=dept.value,
                            access_level=f"level_1_{dept.value}"
                        )
                        dept_results.append(result)
                        response.all_results.append(result)
                    
                    response.department_results[dept.value] = dept_results
        
        # Step 3: Re-rank all results by relevance
        response.all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        response.all_results = response.all_results[:n_results]
        response.total_count = len(response.all_results)
        
        return response
    
    def retrieve_from_department(
        self,
        query: str,
        department: Department,
        user_role: UserRole,
        n_results: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve from a specific department only.
        
        Args:
            query: Search query
            department: Target department
            user_role: User's role for access check
            n_results: Maximum results
            
        Returns:
            List of RetrievalResult objects
        """
        # Check access
        accessible = self.get_accessible_departments(user_role)
        if department not in accessible:
            return []
        
        dept_raw = self.level_contexts.query_department(
            department=department,
            query_text=query,
            n_results=n_results
        )
        
        results = []
        for dr in dept_raw:
            result = RetrievalResult(
                content=dr.content,
                metadata=dr.metadata,
                relevance_score=dr.relevance_score,
                document_id=dr.document_id,
                chunk_index=dr.chunk_index,
                source=department.value,
                access_level=f"level_1_{department.value}"
            )
            results.append(result)
        
        return results
    
    def format_context_for_llm(
        self,
        retrieval_response: RetrievalResponse,
        max_tokens: int = 3000,
        include_sources: bool = True
    ) -> str:
        """
        Format retrieval results into context string for LLM.
        
        Args:
            retrieval_response: Retrieval results
            max_tokens: Maximum approximate tokens
            include_sources: Whether to include source citations
            
        Returns:
            Formatted context string
        """
        if not retrieval_response.all_results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        current_tokens = 0
        tokens_per_char = 0.25
        
        for i, result in enumerate(retrieval_response.all_results):
            # Build result text
            if include_sources:
                source_info = f"[Source: {result.metadata.get('file_name', 'Unknown')} | {result.source.upper()}]"
                result_text = f"{source_info}\n{result.content}\n"
            else:
                result_text = f"{result.content}\n"
            
            # Check token limit
            estimated_tokens = int(len(result_text) * tokens_per_char)
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            context_parts.append(result_text)
            current_tokens += estimated_tokens
        
        return "\n---\n".join(context_parts)
    
    def get_retrieval_stats(self, user_role: UserRole) -> Dict[str, Any]:
        """
        Get statistics about accessible knowledge bases.
        
        Args:
            user_role: User's role
            
        Returns:
            Stats dict
        """
        public_only = self.is_public_only_user(user_role)
        
        stats = {
            "role": user_role.value,
            "public_only": public_only,
            "main_context_count": self.main_context.get_document_count(public_only),
            "department_counts": {},
            "accessible_departments": []
        }
        
        if not public_only:
            accessible = self.get_accessible_departments(user_role)
            stats["accessible_departments"] = [d.value for d in accessible]
            
            for dept in accessible:
                count = self.level_contexts.get_department_document_count(dept)
                stats["department_counts"][dept.value] = count
        
        return stats
