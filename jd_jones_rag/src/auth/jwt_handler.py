"""
JWT Handler
Handles JWT token creation and validation for API authentication.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from functools import wraps

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from src.config.settings import settings


# Security scheme
security = HTTPBearer()


def create_jwt_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT token.
    
    Args:
        data: Payload data to encode
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        Decoded payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User data from token payload
    """
    token = credentials.credentials
    payload = decode_jwt_token(token)
    
    # Extract user info
    user_data = {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "employee"),
        "department": payload.get("department"),
        "permissions": payload.get("permissions", []),
        "exp": payload.get("exp"),
        "iat": payload.get("iat")
    }
    
    if not user_data["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_data


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise.
    
    Args:
        credentials: Optional HTTP Bearer credentials
        
    Returns:
        User data or None
    """
    if not credentials:
        return None
    
    try:
        return decode_jwt_token(credentials.credentials)
    except HTTPException:
        return None


def require_auth(func):
    """Decorator to require authentication."""
    @wraps(func)
    async def wrapper(*args, current_user: Dict[str, Any] = Depends(get_current_user), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper


def require_role(allowed_roles: list):
    """
    Decorator factory to require specific roles.
    
    Args:
        allowed_roles: List of allowed role names
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: Dict[str, Any] = Depends(get_current_user), **kwargs):
            if current_user.get("role") not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {current_user.get('role')} not authorized. Requires: {allowed_roles}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


class JWTBearer:
    """
    Custom JWT Bearer authentication.
    Provides more control over token validation.
    """
    
    def __init__(self, auto_error: bool = True):
        self.auto_error = auto_error
    
    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Optional[Dict[str, Any]]:
        if credentials:
            if credentials.scheme != "Bearer":
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Invalid authentication scheme"
                    )
                return None
            
            payload = decode_jwt_token(credentials.credentials)
            return payload
        
        if self.auto_error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization credentials"
            )
        return None
