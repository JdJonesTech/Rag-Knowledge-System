"""Authentication and authorization module."""

from src.auth.authentication import (
    User,
    create_access_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    authenticate_user
)
from src.auth.authorization import (
    Permission,
    AuthorizationService,
    require_permission,
    check_department_access
)
from src.auth.document_access import (
    AccessLevel,
    DataClassification,
    DocumentAccessPolicy,
    UserAccessContext,
    DocumentAccessController,
    document_access_controller,
    filter_results_by_access
)
from src.auth.jwt_handler import (
    create_jwt_token,
    decode_jwt_token,
    get_current_user as get_jwt_user,
    get_optional_user,
    JWTBearer
)
from src.auth.rbac import (
    UserRole,
    Permission as RBACPermission,
    check_permission,
    check_any_permission,
    check_all_permissions,
    get_role_permissions,
    require_permission as require_rbac_permission,
    require_role,
    RBACMiddleware
)

__all__ = [
    # Authentication
    "User",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "authenticate_user",
    # Authorization
    "Permission",
    "AuthorizationService",
    "require_permission",
    "check_department_access",
    # Document Access Control
    "AccessLevel",
    "DataClassification",
    "DocumentAccessPolicy",
    "UserAccessContext",
    "DocumentAccessController",
    "document_access_controller",
    "filter_results_by_access",
    # JWT
    "create_jwt_token",
    "decode_jwt_token",
    "get_jwt_user",
    "get_optional_user",
    "JWTBearer",
    # RBAC
    "UserRole",
    "RBACPermission",
    "check_permission",
    "check_any_permission",
    "check_all_permissions",
    "get_role_permissions",
    "require_rbac_permission",
    "require_role",
    "RBACMiddleware"
]
