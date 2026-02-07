"""
Role-Based Authorization System.
Handles permission checking and access control.
"""

from enum import Enum
from typing import List, Set, Optional, Callable
from functools import wraps

from fastapi import HTTPException, status, Depends

from src.auth.authentication import User, get_current_active_user
from src.knowledge_base.level_contexts import Department


class Permission(str, Enum):
    """System permissions."""
    # Main context permissions
    VIEW_MAIN_CONTEXT = "view_main_context"
    VIEW_PUBLIC_ONLY = "view_public_only"
    
    # Department context permissions
    VIEW_SALES_CONTEXT = "view_sales_context"
    VIEW_PRODUCTION_CONTEXT = "view_production_context"
    VIEW_ENGINEERING_CONTEXT = "view_engineering_context"
    VIEW_CUSTOMER_SERVICE_CONTEXT = "view_customer_service_context"
    VIEW_MANAGEMENT_CONTEXT = "view_management_context"
    VIEW_ALL_DEPARTMENTS = "view_all_departments"
    
    # Data access permissions
    ACCESS_CUSTOMER_DATA = "access_customer_data"
    ACCESS_FINANCIAL_DATA = "access_financial_data"
    ACCESS_CONFIDENTIAL = "access_confidential"
    
    # Action permissions
    SEARCH_DOCUMENTS = "search_documents"
    CHAT_WITH_AGENT = "chat_with_agent"
    SUBMIT_INQUIRIES = "submit_inquiries"
    GENERATE_QUOTES = "generate_quotes"
    
    # Admin permissions
    MANAGE_DOCUMENTS = "manage_documents"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"


# Role to permission mapping
ROLE_PERMISSIONS: dict[str, Set[Permission]] = {
    "employee": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
    },
    "sales_rep": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_SALES_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
        Permission.GENERATE_QUOTES,
    },
    "sales_manager": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_SALES_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
        Permission.GENERATE_QUOTES,
        Permission.MANAGE_DOCUMENTS,
    },
    "production_worker": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_PRODUCTION_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
    },
    "production_supervisor": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_PRODUCTION_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.MANAGE_DOCUMENTS,
    },
    "engineer": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_ENGINEERING_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
    },
    "engineering_manager": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_ENGINEERING_CONTEXT,
        Permission.VIEW_PRODUCTION_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.MANAGE_DOCUMENTS,
    },
    "customer_service_rep": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_CUSTOMER_SERVICE_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
    },
    "customer_service_manager": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_CUSTOMER_SERVICE_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
        Permission.MANAGE_DOCUMENTS,
    },
    "manager": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_ALL_DEPARTMENTS,
        Permission.VIEW_SALES_CONTEXT,
        Permission.VIEW_PRODUCTION_CONTEXT,
        Permission.VIEW_ENGINEERING_CONTEXT,
        Permission.VIEW_CUSTOMER_SERVICE_CONTEXT,
        Permission.VIEW_MANAGEMENT_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
        Permission.GENERATE_QUOTES,
        Permission.MANAGE_DOCUMENTS,
    },
    "executive": {
        Permission.VIEW_MAIN_CONTEXT,
        Permission.VIEW_ALL_DEPARTMENTS,
        Permission.VIEW_SALES_CONTEXT,
        Permission.VIEW_PRODUCTION_CONTEXT,
        Permission.VIEW_ENGINEERING_CONTEXT,
        Permission.VIEW_CUSTOMER_SERVICE_CONTEXT,
        Permission.VIEW_MANAGEMENT_CONTEXT,
        Permission.SEARCH_DOCUMENTS,
        Permission.CHAT_WITH_AGENT,
        Permission.ACCESS_CUSTOMER_DATA,
        Permission.ACCESS_FINANCIAL_DATA,
        Permission.ACCESS_CONFIDENTIAL,
        Permission.GENERATE_QUOTES,
        Permission.MANAGE_DOCUMENTS,
        Permission.MANAGE_USERS,
        Permission.ADMIN_ACCESS,
    },
    "external_customer": {
        Permission.VIEW_PUBLIC_ONLY,
        Permission.SUBMIT_INQUIRIES,
    },
}


# Department to permission mapping
DEPARTMENT_VIEW_PERMISSIONS = {
    Department.SALES: Permission.VIEW_SALES_CONTEXT,
    Department.PRODUCTION: Permission.VIEW_PRODUCTION_CONTEXT,
    Department.ENGINEERING: Permission.VIEW_ENGINEERING_CONTEXT,
    Department.CUSTOMER_SERVICE: Permission.VIEW_CUSTOMER_SERVICE_CONTEXT,
    Department.MANAGEMENT: Permission.VIEW_MANAGEMENT_CONTEXT,
}


class AuthorizationService:
    """Service for checking user permissions and access rights."""
    
    @staticmethod
    def get_user_permissions(user: User) -> Set[Permission]:
        """
        Get all permissions for a user based on their role.
        
        Args:
            user: User object
            
        Returns:
            Set of Permission enums
        """
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        
        # Add basic permissions for all active internal users
        if user.is_internal and user.is_active:
            role_perms = role_perms | {Permission.VIEW_MAIN_CONTEXT}
        
        return role_perms
    
    @staticmethod
    def check_permission(user: User, required_permission: Permission) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User object
            required_permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user_permissions = AuthorizationService.get_user_permissions(user)
        return required_permission in user_permissions
    
    @staticmethod
    def check_any_permission(user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions.
        
        Args:
            user: User object
            permissions: List of permissions to check
            
        Returns:
            True if user has at least one permission
        """
        user_permissions = AuthorizationService.get_user_permissions(user)
        return any(p in user_permissions for p in permissions)
    
    @staticmethod
    def check_all_permissions(user: User, permissions: List[Permission]) -> bool:
        """
        Check if user has all of the specified permissions.
        
        Args:
            user: User object
            permissions: List of permissions to check
            
        Returns:
            True if user has all permissions
        """
        user_permissions = AuthorizationService.get_user_permissions(user)
        return all(p in user_permissions for p in permissions)
    
    @staticmethod
    def get_accessible_departments(user: User) -> List[Department]:
        """
        Get list of departments a user can access.
        
        Args:
            user: User object
            
        Returns:
            List of accessible departments
        """
        user_permissions = AuthorizationService.get_user_permissions(user)
        
        # If user has VIEW_ALL_DEPARTMENTS, return all
        if Permission.VIEW_ALL_DEPARTMENTS in user_permissions:
            return list(Department)
        
        # Otherwise, check individual department permissions
        accessible = []
        for dept, perm in DEPARTMENT_VIEW_PERMISSIONS.items():
            if perm in user_permissions:
                accessible.append(dept)
        
        return accessible
    
    @staticmethod
    def can_access_department(user: User, department: Department) -> bool:
        """
        Check if user can access a specific department's data.
        
        Args:
            user: User object
            department: Department to check
            
        Returns:
            True if user can access department
        """
        accessible = AuthorizationService.get_accessible_departments(user)
        return department in accessible
    
    @staticmethod
    def can_access_public_only(user: User) -> bool:
        """
        Check if user should only see public content.
        
        Args:
            user: User object
            
        Returns:
            True if user is limited to public content
        """
        user_permissions = AuthorizationService.get_user_permissions(user)
        return (
            Permission.VIEW_PUBLIC_ONLY in user_permissions and
            Permission.VIEW_MAIN_CONTEXT not in user_permissions
        )


def require_permission(required_permission: Permission):
    """
    Decorator to require a specific permission for an endpoint.
    
    Args:
        required_permission: Permission required to access endpoint
        
    Returns:
        Decorator function
    """
    async def permission_dependency(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not AuthorizationService.check_permission(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {required_permission.value} required"
            )
        return current_user
    
    return permission_dependency


def require_any_permission(permissions: List[Permission]):
    """
    Decorator to require any of the specified permissions.
    
    Args:
        permissions: List of acceptable permissions
        
    Returns:
        Decorator function
    """
    async def permission_dependency(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not AuthorizationService.check_any_permission(current_user, permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied: insufficient privileges"
            )
        return current_user
    
    return permission_dependency


def check_department_access(department: Department):
    """
    Decorator to require access to a specific department.
    
    Args:
        department: Department requiring access
        
    Returns:
        Decorator function
    """
    async def department_dependency(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not AuthorizationService.can_access_department(current_user, department):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {department.value} department"
            )
        return current_user
    
    return department_dependency
