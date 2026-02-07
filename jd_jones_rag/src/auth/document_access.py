"""
Document-Level Access Control
Ensures agents can only retrieve data that the user is authorized to see.
Integrates with the RBAC system for fine-grained document filtering.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class AccessLevel(str, Enum):
    """Document access levels."""
    PUBLIC = "public"                    # Anyone can access
    INTERNAL = "internal"                # Any employee
    DEPARTMENT = "department"            # Specific department only
    CONFIDENTIAL = "confidential"        # Manager+ or specific roles
    RESTRICTED = "restricted"            # Executive or specific named users
    TOP_SECRET = "top_secret"            # Named users only


class DataClassification(str, Enum):
    """Data classification types for compliance."""
    GENERAL = "general"
    PII = "pii"                          # Personal Identifiable Information
    FINANCIAL = "financial"              # Financial data
    LEGAL = "legal"                      # Legal documents
    TRADE_SECRET = "trade_secret"        # Proprietary information
    CUSTOMER_DATA = "customer_data"      # Customer information


@dataclass
class DocumentAccessPolicy:
    """Access policy for a document."""
    document_id: str
    access_level: AccessLevel
    classifications: List[DataClassification] = field(default_factory=list)
    allowed_departments: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)
    denied_users: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "access_level": self.access_level.value,
            "classifications": [c.value for c in self.classifications],
            "allowed_departments": self.allowed_departments,
            "allowed_roles": self.allowed_roles,
            "allowed_users": self.allowed_users,
            "denied_users": self.denied_users
        }


@dataclass
class UserAccessContext:
    """User's access context for filtering."""
    user_id: str
    role: str
    department: str
    permissions: Set[str] = field(default_factory=set)
    clearance_level: AccessLevel = AccessLevel.INTERNAL
    data_classifications_allowed: Set[DataClassification] = field(default_factory=set)
    
    @classmethod
    def from_user(cls, user: Dict[str, Any]) -> "UserAccessContext":
        """Create from user dict."""
        role = user.get("role", "employee")
        
        # Determine clearance level based on role
        clearance_mapping = {
            "employee": AccessLevel.INTERNAL,
            "sales_rep": AccessLevel.DEPARTMENT,
            "sales_manager": AccessLevel.CONFIDENTIAL,
            "production_worker": AccessLevel.INTERNAL,
            "production_supervisor": AccessLevel.DEPARTMENT,
            "engineer": AccessLevel.DEPARTMENT,
            "manager": AccessLevel.CONFIDENTIAL,
            "executive": AccessLevel.RESTRICTED,
            "admin": AccessLevel.TOP_SECRET,
            "external_customer": AccessLevel.PUBLIC
        }
        
        # Determine allowed data classifications
        classification_mapping = {
            "employee": {DataClassification.GENERAL},
            "sales_rep": {DataClassification.GENERAL, DataClassification.CUSTOMER_DATA},
            "sales_manager": {DataClassification.GENERAL, DataClassification.CUSTOMER_DATA, DataClassification.FINANCIAL},
            "finance": {DataClassification.GENERAL, DataClassification.FINANCIAL},
            "legal": {DataClassification.GENERAL, DataClassification.LEGAL, DataClassification.PII},
            "hr": {DataClassification.GENERAL, DataClassification.PII},
            "manager": {DataClassification.GENERAL, DataClassification.CUSTOMER_DATA, DataClassification.FINANCIAL},
            "executive": {DataClassification.GENERAL, DataClassification.CUSTOMER_DATA, DataClassification.FINANCIAL, DataClassification.LEGAL, DataClassification.TRADE_SECRET},
            "admin": set(DataClassification),
            "external_customer": {DataClassification.GENERAL}
        }
        
        return cls(
            user_id=user.get("user_id", user.get("sub", "anonymous")),
            role=role,
            department=user.get("department", "general"),
            permissions=set(user.get("permissions", [])),
            clearance_level=clearance_mapping.get(role, AccessLevel.INTERNAL),
            data_classifications_allowed=classification_mapping.get(role, {DataClassification.GENERAL})
        )


class DocumentAccessController:
    """
    Controls document-level access based on RBAC.
    
    This integrates with the agentic system to ensure:
    1. Agents only retrieve documents the user can access
    2. Document metadata is checked against user permissions
    3. Data classifications are respected
    4. Audit trail is maintained
    """
    
    # Default policies by document type
    DEFAULT_POLICIES = {
        "datasheet": {
            "access_level": AccessLevel.PUBLIC,
            "classifications": [DataClassification.GENERAL]
        },
        "price_list": {
            "access_level": AccessLevel.DEPARTMENT,
            "allowed_departments": ["sales"],
            "classifications": [DataClassification.FINANCIAL]
        },
        "customer_data": {
            "access_level": AccessLevel.CONFIDENTIAL,
            "allowed_roles": ["sales_rep", "sales_manager", "manager", "executive"],
            "classifications": [DataClassification.CUSTOMER_DATA, DataClassification.PII]
        },
        "financial_report": {
            "access_level": AccessLevel.CONFIDENTIAL,
            "allowed_departments": ["finance", "executive"],
            "classifications": [DataClassification.FINANCIAL]
        },
        "legal_document": {
            "access_level": AccessLevel.RESTRICTED,
            "allowed_departments": ["legal", "executive"],
            "classifications": [DataClassification.LEGAL]
        },
        "employee_record": {
            "access_level": AccessLevel.RESTRICTED,
            "allowed_departments": ["hr", "executive"],
            "classifications": [DataClassification.PII]
        },
        "trade_secret": {
            "access_level": AccessLevel.TOP_SECRET,
            "allowed_roles": ["executive", "admin"],
            "classifications": [DataClassification.TRADE_SECRET]
        }
    }
    
    def __init__(self):
        """Initialize access controller."""
        # Document policies cache
        self.policies: Dict[str, DocumentAccessPolicy] = {}
        
        # Access audit log
        self.audit_log: List[Dict[str, Any]] = []
    
    def set_document_policy(
        self,
        document_id: str,
        policy: DocumentAccessPolicy
    ):
        """Set access policy for a document."""
        self.policies[document_id] = policy
    
    def get_document_policy(
        self,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentAccessPolicy:
        """
        Get access policy for a document.
        Creates default policy based on metadata if not explicitly set.
        """
        if document_id in self.policies:
            return self.policies[document_id]
        
        # Create policy from metadata
        if document_metadata:
            doc_type = document_metadata.get("document_type", "general")
            default = self.DEFAULT_POLICIES.get(doc_type, {})
            
            return DocumentAccessPolicy(
                document_id=document_id,
                access_level=AccessLevel(default.get("access_level", "internal")),
                classifications=[DataClassification(c) for c in default.get("classifications", ["general"])],
                allowed_departments=default.get("allowed_departments", []),
                allowed_roles=default.get("allowed_roles", [])
            )
        
        # Default internal access
        return DocumentAccessPolicy(
            document_id=document_id,
            access_level=AccessLevel.INTERNAL,
            classifications=[DataClassification.GENERAL]
        )
    
    def check_access(
        self,
        user_context: UserAccessContext,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if user can access a document.
        
        Args:
            user_context: User's access context
            document_id: Document to check
            document_metadata: Optional document metadata
            
        Returns:
            Dict with 'allowed' bool and 'reason' string
        """
        policy = self.get_document_policy(document_id, document_metadata)
        
        # Check if user is explicitly denied
        if user_context.user_id in policy.denied_users:
            self._log_access(user_context, document_id, False, "User explicitly denied")
            return {"allowed": False, "reason": "Access explicitly denied"}
        
        # Check if user is explicitly allowed
        if user_context.user_id in policy.allowed_users:
            self._log_access(user_context, document_id, True, "User explicitly allowed")
            return {"allowed": True, "reason": "User explicitly allowed"}
        
        # Check expiration
        if policy.expires_at and datetime.now() > policy.expires_at:
            self._log_access(user_context, document_id, False, "Policy expired")
            return {"allowed": False, "reason": "Document access has expired"}
        
        # Check clearance level
        clearance_order = [
            AccessLevel.PUBLIC,
            AccessLevel.INTERNAL,
            AccessLevel.DEPARTMENT,
            AccessLevel.CONFIDENTIAL,
            AccessLevel.RESTRICTED,
            AccessLevel.TOP_SECRET
        ]
        
        user_clearance_idx = clearance_order.index(user_context.clearance_level)
        doc_clearance_idx = clearance_order.index(policy.access_level)
        
        if user_clearance_idx < doc_clearance_idx:
            self._log_access(user_context, document_id, False, "Insufficient clearance")
            return {"allowed": False, "reason": f"Requires {policy.access_level.value} clearance"}
        
        # Check data classifications
        for classification in policy.classifications:
            if classification not in user_context.data_classifications_allowed:
                self._log_access(user_context, document_id, False, f"Missing {classification.value} access")
                return {"allowed": False, "reason": f"Requires {classification.value} data access"}
        
        # Check department restrictions
        if policy.allowed_departments:
            if user_context.department not in policy.allowed_departments:
                # Check if user has cross-department access
                if user_context.clearance_level not in [AccessLevel.CONFIDENTIAL, AccessLevel.RESTRICTED, AccessLevel.TOP_SECRET]:
                    self._log_access(user_context, document_id, False, "Department restricted")
                    return {"allowed": False, "reason": f"Restricted to departments: {policy.allowed_departments}"}
        
        # Check role restrictions
        if policy.allowed_roles:
            if user_context.role not in policy.allowed_roles:
                self._log_access(user_context, document_id, False, "Role restricted")
                return {"allowed": False, "reason": f"Restricted to roles: {policy.allowed_roles}"}
        
        # Access granted
        self._log_access(user_context, document_id, True, "Access granted")
        return {"allowed": True, "reason": "Access granted"}
    
    def filter_documents(
        self,
        user_context: UserAccessContext,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of documents based on user access.
        
        Args:
            user_context: User's access context
            documents: List of documents with metadata
            
        Returns:
            Filtered list of accessible documents
        """
        accessible = []
        
        for doc in documents:
            doc_id = doc.get("id", doc.get("document_id", str(hash(str(doc)))))
            metadata = doc.get("metadata", doc)
            
            access_result = self.check_access(user_context, doc_id, metadata)
            
            if access_result["allowed"]:
                accessible.append(doc)
        
        return accessible
    
    def filter_search_results(
        self,
        user_context: UserAccessContext,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter search results based on user access.
        Redacts sensitive fields if partial access allowed.
        
        Args:
            user_context: User's access context
            results: Search results
            
        Returns:
            Filtered and potentially redacted results
        """
        filtered = []
        
        for result in results:
            doc_id = result.get("id", result.get("document_id", ""))
            metadata = result.get("metadata", {})
            
            access_result = self.check_access(user_context, doc_id, metadata)
            
            if access_result["allowed"]:
                # Check if we need to redact any fields
                redacted_result = self._redact_sensitive_fields(user_context, result)
                filtered.append(redacted_result)
        
        return filtered
    
    def _redact_sensitive_fields(
        self,
        user_context: UserAccessContext,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Redact sensitive fields based on user permissions."""
        redacted = result.copy()
        content = redacted.get("content", "")
        
        # Check if user can see PII
        if DataClassification.PII not in user_context.data_classifications_allowed:
            # Redact email addresses
            import re
            content = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL REDACTED]', content)
            # Redact phone numbers
            content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', content)
            # Redact SSN patterns
            content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', content)
        
        # Check if user can see financial data
        if DataClassification.FINANCIAL not in user_context.data_classifications_allowed:
            # Redact dollar amounts
            content = re.sub(r'\$[\d,]+\.?\d*', '[AMOUNT REDACTED]', content)
            # Redact percentages in financial context
            if "price" in content.lower() or "discount" in content.lower():
                content = re.sub(r'\d+\.?\d*%', '[PERCENTAGE REDACTED]', content)
        
        redacted["content"] = content
        return redacted
    
    def _log_access(
        self,
        user_context: UserAccessContext,
        document_id: str,
        allowed: bool,
        reason: str
    ):
        """Log access attempt for audit."""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_context.user_id,
            "role": user_context.role,
            "department": user_context.department,
            "document_id": document_id,
            "allowed": allowed,
            "reason": reason
        })
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        log = self.audit_log
        
        if user_id:
            log = [e for e in log if e["user_id"] == user_id]
        if document_id:
            log = [e for e in log if e["document_id"] == document_id]
        
        return log[-limit:]


# Global instance
document_access_controller = DocumentAccessController()


def filter_results_by_access(
    user: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convenience function to filter search results by user access.
    
    Usage in agentic tools:
        results = vector_search(query)
        filtered = filter_results_by_access(current_user, results)
    """
    user_context = UserAccessContext.from_user(user)
    return document_access_controller.filter_search_results(user_context, results)
