"""
Level Context Databases - Level 1+ (Department-Specific) Knowledge Bases.
Stores and retrieves department-specific information with role-based access.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.data_ingestion.vector_store import VectorStoreManager, SearchResult
from src.data_ingestion.embedding_generator import EmbeddedDocument
from src.config.settings import settings


class Department(str, Enum):
    """Department identifiers."""
    SALES = "sales"
    PRODUCTION = "production"
    ENGINEERING = "engineering"
    CUSTOMER_SERVICE = "customer_service"
    MANAGEMENT = "management"


@dataclass
class DepartmentContextResult:
    """Result from department context query."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    document_id: str
    chunk_index: int
    department: str
    document_type: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "department": self.department,
            "document_type": self.document_type
        }


class LevelContextDatabase:
    """
    Level 1+ (Department Context) Knowledge Base.
    
    Manages department-specific collections:
    - Sales: Pricing, customers, commissions, competitors
    - Production: Work instructions, machine manuals, safety
    - Engineering: Designs, material certs, test reports
    - Customer Service: Support scripts, escalation procedures
    """
    
    # Collection name prefix
    COLLECTION_PREFIX = "jd_jones"
    
    # Department-specific document types
    DEPARTMENT_DOC_TYPES = {
        Department.SALES: [
            "pricing_guide",
            "customer_data",
            "commission_rules",
            "competitor_analysis",
            "sales_playbook",
            "quote_template"
        ],
        Department.PRODUCTION: [
            "work_instruction",
            "machine_manual",
            "safety_protocol",
            "maintenance_schedule",
            "quality_procedure",
            "equipment_spec"
        ],
        Department.ENGINEERING: [
            "design_document",
            "material_certificate",
            "test_report",
            "compliance_document",
            "technical_drawing",
            "specification"
        ],
        Department.CUSTOMER_SERVICE: [
            "support_script",
            "escalation_procedure",
            "return_policy",
            "warranty_info",
            "troubleshooting_guide",
            "faq"
        ],
        Department.MANAGEMENT: [
            "performance_report",
            "budget_document",
            "strategic_plan",
            "meeting_notes",
            "policy_draft"
        ]
    }
    
    def __init__(self):
        """Initialize level context database."""
        self.vector_store = VectorStoreManager()
        self._initialize_collections()
    
    def _get_collection_name(self, department: Department) -> str:
        """Get collection name for a department."""
        return f"{self.COLLECTION_PREFIX}_{department.value}"
    
    def _initialize_collections(self):
        """Create collections for all departments."""
        for department in Department:
            collection_name = self._get_collection_name(department)
            doc_types = self.DEPARTMENT_DOC_TYPES.get(department, [])
            
            self.vector_store.create_collection(
                collection_name,
                metadata={
                    "description": f"JD Jones {department.value} department knowledge base",
                    "level": "1",
                    "department": department.value,
                    "document_types": ",".join(doc_types)
                }
            )
    
    def add_department_documents(
        self,
        department: Department,
        documents: List[EmbeddedDocument]
    ) -> int:
        """
        Add documents to a department's knowledge base.
        
        Args:
            department: Target department
            documents: List of embedded documents
            
        Returns:
            Number of documents added
        """
        collection_name = self._get_collection_name(department)
        
        # Ensure department is set in metadata
        for doc in documents:
            doc.metadata["department"] = department.value
        
        return self.vector_store.add_documents(
            collection_name,
            documents,
            upsert=True
        )
    
    def query_department(
        self,
        department: Department,
        query_text: str,
        n_results: int = 10,
        document_type: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DepartmentContextResult]:
        """
        Query a specific department's knowledge base.
        
        Args:
            department: Department to query
            query_text: Search query
            n_results: Maximum results
            document_type: Filter by document type
            filter_metadata: Additional metadata filters
            
        Returns:
            List of DepartmentContextResult objects
        """
        collection_name = self._get_collection_name(department)
        
        # Build where filter
        where = filter_metadata.copy() if filter_metadata else {}
        
        if document_type:
            where["document_type"] = document_type
        
        # Perform search
        search_results = self.vector_store.search(
            collection_name=collection_name,
            query_text=query_text,
            n_results=n_results,
            where=where if where else None
        )
        
        # Convert to DepartmentContextResult
        results = []
        for sr in search_results:
            result = DepartmentContextResult(
                content=sr.content,
                metadata=sr.metadata,
                relevance_score=sr.relevance_score,
                document_id=sr.document_id,
                chunk_index=sr.chunk_index,
                department=department.value,
                document_type=sr.metadata.get("document_type")
            )
            results.append(result)
        
        return results
    
    def query_multi_department(
        self,
        query_text: str,
        departments: List[Department],
        n_results_per_dept: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[DepartmentContextResult]]:
        """
        Query multiple departments simultaneously.
        
        Args:
            query_text: Search query
            departments: List of departments to query
            n_results_per_dept: Max results per department
            filter_metadata: Additional filters
            
        Returns:
            Dict mapping department names to results
        """
        results = {}
        
        for department in departments:
            dept_results = self.query_department(
                department=department,
                query_text=query_text,
                n_results=n_results_per_dept,
                filter_metadata=filter_metadata
            )
            results[department.value] = dept_results
        
        return results
    
    def query_all_accessible(
        self,
        query_text: str,
        accessible_departments: List[Department],
        n_results_total: int = 10
    ) -> List[DepartmentContextResult]:
        """
        Query all accessible departments and return merged results.
        
        Args:
            query_text: Search query
            accessible_departments: Departments user can access
            n_results_total: Total max results
            
        Returns:
            Merged and ranked list of results
        """
        # Query each department
        all_results = []
        results_per_dept = max(3, n_results_total // len(accessible_departments))
        
        for department in accessible_departments:
            dept_results = self.query_department(
                department=department,
                query_text=query_text,
                n_results=results_per_dept
            )
            all_results.extend(dept_results)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Return top N
        return all_results[:n_results_total]
    
    def get_department_document_count(self, department: Department) -> int:
        """
        Get document count for a department.
        
        Args:
            department: Department to count
            
        Returns:
            Document count
        """
        collection_name = self._get_collection_name(department)
        return self.vector_store.get_collection_count(collection_name)
    
    def get_all_document_counts(self) -> Dict[str, int]:
        """
        Get document counts for all departments.
        
        Returns:
            Dict mapping department names to counts
        """
        counts = {}
        for department in Department:
            counts[department.value] = self.get_department_document_count(department)
        return counts
    
    def delete_department_document(
        self,
        department: Department,
        document_id: str
    ) -> bool:
        """
        Delete a document from a department's knowledge base.
        
        Args:
            department: Department
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(department)
        return self.vector_store.delete_documents(
            collection_name,
            where={"document_id": document_id}
        )
    
    def get_document_types(self, department: Department) -> List[str]:
        """
        Get supported document types for a department.
        
        Args:
            department: Department
            
        Returns:
            List of document type strings
        """
        return self.DEPARTMENT_DOC_TYPES.get(department, [])
    
    def format_context_for_llm(
        self,
        results: List[DepartmentContextResult],
        max_tokens: int = 2000
    ) -> str:
        """
        Format department results for LLM context.
        
        Args:
            results: List of search results
            max_tokens: Maximum token limit
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant department information found."
        
        context_parts = []
        current_tokens = 0
        tokens_per_char = 0.25
        
        # Group by department for clarity
        by_department = {}
        for result in results:
            dept = result.department
            if dept not in by_department:
                by_department[dept] = []
            by_department[dept].append(result)
        
        for dept, dept_results in by_department.items():
            dept_header = f"=== {dept.upper()} DEPARTMENT ===\n"
            context_parts.append(dept_header)
            current_tokens += int(len(dept_header) * tokens_per_char)
            
            for result in dept_results:
                result_text = f"[{result.metadata.get('file_name', 'Unknown')}]\n{result.content}\n"
                estimated_tokens = int(len(result_text) * tokens_per_char)
                
                if current_tokens + estimated_tokens > max_tokens:
                    break
                
                context_parts.append(result_text)
                current_tokens += estimated_tokens
        
        return "\n".join(context_parts)
