"""
Role-Based Access Control (RBAC)
Provides role and permission checking for API endpoints.
"""

from typing import Dict, Any, List, Set, Optional, Callable
from enum import Enum
from functools import wraps

from fastapi import HTTPException, status, Depends

from src.auth.jwt_handler import get_current_user


class UserRole(str, Enum):
    """User roles in the system."""
    # Basic roles
    GUEST = "guest"
    EMPLOYEE = "employee"
    
    # Department-specific roles
    SALES_REP = "sales_rep"
    SALES_MANAGER = "sales_manager"
    ENGINEER = "engineer"
    ENGINEERING_MANAGER = "engineering_manager"
    PRODUCTION_WORKER = "production_worker"
    PRODUCTION_SUPERVISOR = "production_supervisor"
    CUSTOMER_SERVICE = "customer_service"
    
    # Management roles
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"
    
    # System roles
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    
    # External roles
    EXTERNAL_CUSTOMER = "external_customer"
    PARTNER = "partner"


class Permission(str, Enum):
    """System permissions."""
    # Read permissions
    READ_PUBLIC = "read:public"
    READ_INTERNAL = "read:internal"
    READ_CONFIDENTIAL = "read:confidential"
    READ_RESTRICTED = "read:restricted"
    
    # Department read permissions
    READ_SALES = "read:sales"
    READ_ENGINEERING = "read:engineering"
    READ_PRODUCTION = "read:production"
    READ_FINANCE = "read:finance"
    READ_HR = "read:hr"
    
    # Write permissions
    WRITE_DOCUMENTS = "write:documents"
    WRITE_CUSTOMERS = "write:customers"
    WRITE_ORDERS = "write:orders"
    WRITE_PRODUCTS = "write:products"
    
    # Action permissions
    EXECUTE_QUERIES = "execute:queries"
    EXECUTE_TOOLS = "execute:tools"
    APPROVE_ACTIONS = "approve:actions"
    MANAGE_USERS = "manage:users"
    MANAGE_SYSTEM = "manage:system"
    
    # Agentic permissions
    USE_AGENTIC = "use:agentic"
    USE_MULTI_AGENT = "use:multi_agent"
    VIEW_TRACES = "view:traces"
    MANAGE_APPROVALS = "manage:approvals"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.GUEST: {
        Permission.READ_PUBLIC,
    },
    
    UserRole.EMPLOYEE: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.EXECUTE_QUERIES,
        Permission.USE_AGENTIC,
    },
    
    UserRole.SALES_REP: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_SALES,
        Permission.WRITE_CUSTOMERS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.USE_AGENTIC,
    },
    
    UserRole.SALES_MANAGER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_SALES,
        Permission.READ_CONFIDENTIAL,
        Permission.WRITE_CUSTOMERS,
        Permission.WRITE_ORDERS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.APPROVE_ACTIONS,
        Permission.USE_AGENTIC,
        Permission.USE_MULTI_AGENT,
        Permission.VIEW_TRACES,
    },
    
    UserRole.ENGINEER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_ENGINEERING,
        Permission.WRITE_DOCUMENTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.USE_AGENTIC,
    },
    
    UserRole.ENGINEERING_MANAGER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_ENGINEERING,
        Permission.READ_CONFIDENTIAL,
        Permission.WRITE_DOCUMENTS,
        Permission.WRITE_PRODUCTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.APPROVE_ACTIONS,
        Permission.USE_AGENTIC,
        Permission.USE_MULTI_AGENT,
    },
    
    UserRole.PRODUCTION_WORKER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_PRODUCTION,
        Permission.EXECUTE_QUERIES,
        Permission.USE_AGENTIC,
    },
    
    UserRole.PRODUCTION_SUPERVISOR: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_PRODUCTION,
        Permission.READ_ENGINEERING,
        Permission.WRITE_DOCUMENTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.USE_AGENTIC,
    },
    
    UserRole.CUSTOMER_SERVICE: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_SALES,
        Permission.WRITE_CUSTOMERS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.USE_AGENTIC,
    },
    
    UserRole.MANAGER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_CONFIDENTIAL,
        Permission.WRITE_DOCUMENTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.APPROVE_ACTIONS,
        Permission.USE_AGENTIC,
        Permission.USE_MULTI_AGENT,
        Permission.VIEW_TRACES,
        Permission.MANAGE_APPROVALS,
    },
    
    UserRole.DIRECTOR: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_CONFIDENTIAL,
        Permission.READ_RESTRICTED,
        Permission.READ_SALES,
        Permission.READ_ENGINEERING,
        Permission.READ_PRODUCTION,
        Permission.READ_FINANCE,
        Permission.WRITE_DOCUMENTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.APPROVE_ACTIONS,
        Permission.USE_AGENTIC,
        Permission.USE_MULTI_AGENT,
        Permission.VIEW_TRACES,
        Permission.MANAGE_APPROVALS,
    },
    
    UserRole.EXECUTIVE: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.READ_CONFIDENTIAL,
        Permission.READ_RESTRICTED,
        Permission.READ_SALES,
        Permission.READ_ENGINEERING,
        Permission.READ_PRODUCTION,
        Permission.READ_FINANCE,
        Permission.READ_HR,
        Permission.WRITE_DOCUMENTS,
        Permission.EXECUTE_QUERIES,
        Permission.EXECUTE_TOOLS,
        Permission.APPROVE_ACTIONS,
        Permission.USE_AGENTIC,
        Permission.USE_MULTI_AGENT,
        Permission.VIEW_TRACES,
        Permission.MANAGE_APPROVALS,
    },
    
    UserRole.ADMIN: set(Permission),  # All permissions
    
    UserRole.SUPER_ADMIN: set(Permission),  # All permissions
    
    UserRole.EXTERNAL_CUSTOMER: {
        Permission.READ_PUBLIC,
        Permission.EXECUTE_QUERIES,
    },
    
    UserRole.PARTNER: {
        Permission.READ_PUBLIC,
        Permission.READ_INTERNAL,
        Permission.EXECUTE_QUERIES,
        Permission.USE_AGENTIC,
    },
}


def get_role_permissions(role: str) -> Set[Permission]:
    """
    Get permissions for a role.
    
    Args:
        role: Role name
        
    Returns:
        Set of permissions
    """
    try:
        user_role = UserRole(role)
        return ROLE_PERMISSIONS.get(user_role, set())
    except ValueError:
        return set()


def check_permission(user: Dict[str, Any], permission: Permission) -> bool:
    """
    Check if user has a specific permission.
    
    Args:
        user: User data dict
        permission: Permission to check
        
    Returns:
        True if user has permission
    """
    role = user.get("role", "guest")
    permissions = get_role_permissions(role)
    
    # Also check explicit permissions in user data
    explicit_permissions = set(user.get("permissions", []))
    
    return permission in permissions or permission.value in explicit_permissions


def check_any_permission(user: Dict[str, Any], permissions: List[Permission]) -> bool:
    """
    Check if user has any of the specified permissions.
    
    Args:
        user: User data dict
        permissions: List of permissions to check
        
    Returns:
        True if user has any permission
    """
    return any(check_permission(user, perm) for perm in permissions)


def check_all_permissions(user: Dict[str, Any], permissions: List[Permission]) -> bool:
    """
    Check if user has all specified permissions.
    
    Args:
        user: User data dict
        permissions: List of permissions to check
        
    Returns:
        True if user has all permissions
    """
    return all(check_permission(user, perm) for perm in permissions)


def require_permission(permission: Permission):
    """
    Decorator factory to require a specific permission.
    
    Args:
        permission: Required permission
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(
            *args,
            current_user: Dict[str, Any] = Depends(get_current_user),
            **kwargs
        ):
            if not check_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value} required"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_any_permission(permissions: List[Permission]):
    """
    Decorator factory to require any of the specified permissions.
    
    Args:
        permissions: List of acceptable permissions
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(
            *args,
            current_user: Dict[str, Any] = Depends(get_current_user),
            **kwargs
        ):
            if not check_any_permission(current_user, permissions):
                perm_list = [p.value for p in permissions]
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: one of {perm_list} required"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_role(roles: List[UserRole]):
    """
    Decorator factory to require specific roles.
    
    Args:
        roles: List of acceptable roles
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(
            *args,
            current_user: Dict[str, Any] = Depends(get_current_user),
            **kwargs
        ):
            user_role = current_user.get("role", "guest")
            try:
                role_enum = UserRole(user_role)
                if role_enum not in roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Role {user_role} not authorized"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unknown role: {user_role}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


class RBACMiddleware:
    """
    RBAC middleware for checking permissions on routes.
    """
    
    def __init__(self, required_permission: Optional[Permission] = None):
        self.required_permission = required_permission
    
    async def __call__(
        self,
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        if self.required_permission:
            if not check_permission(current_user, self.required_permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {self.required_permission.value} required"
                )
        return current_user
