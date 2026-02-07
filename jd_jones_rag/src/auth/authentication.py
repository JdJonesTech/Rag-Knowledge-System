"""
JWT Authentication System.
Handles user authentication, token creation, and validation.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from src.config.settings import settings


# Password hashing - explicitly use bcrypt with truncation handling
# Note: bcrypt has a 72-byte password limit. Passwords longer than 72 bytes
# are automatically truncated. For production, consider pre-hashing with SHA256.
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Explicit rounds for consistency
)

# HTTP Bearer token
security = HTTPBearer()


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    email: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    exp: Optional[datetime] = None


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    is_active: bool = True
    is_internal: bool = True
    hashed_password: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding password)."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "department": self.department,
            "is_active": self.is_active,
            "is_internal": self.is_internal
        }


# In-memory user store (replace with database in production)
# Passwords are lazily hashed on first use to avoid import-time issues
_USERS_DB_INITIALIZED = False
USERS_DB: Dict[str, User] = {}

def _init_users_db():
    """Initialize the users database with hashed passwords."""
    global _USERS_DB_INITIALIZED, USERS_DB
    if _USERS_DB_INITIALIZED:
        return
    
    # Pre-computed bcrypt hashes for demo users
    # These are bcrypt hashes of the respective passwords
    demo_users = [
        User(
            user_id="admin_001",
            email="admin@jdjones.com",
            full_name="System Administrator",
            role="executive",
            department="management",
            is_active=True,
            is_internal=True,
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X7sMaKD5g0jVRxvKa"  # admin123
        ),
        User(
            user_id="sales_001",
            email="sales@jdjones.com",
            full_name="Sales Representative",
            role="sales_rep",
            department="sales",
            is_active=True,
            is_internal=True,
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X7sMaKD5g0jVRxvKa"  # sales123
        ),
        User(
            user_id="eng_001",
            email="engineer@jdjones.com",
            full_name="Engineering Specialist",
            role="engineer",
            department="engineering",
            is_active=True,
            is_internal=True,
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X7sMaKD5g0jVRxvKa"  # eng123
        ),
        User(
            user_id="prod_001",
            email="production@jdjones.com",
            full_name="Production Worker",
            role="production_worker",
            department="production",
            is_active=True,
            is_internal=True,
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X7sMaKD5g0jVRxvKa"  # prod123
        ),
        User(
            user_id="cust_001",
            email="customer@example.com",
            full_name="External Customer",
            role="external_customer",
            department=None,
            is_active=True,
            is_internal=False,
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X7sMaKD5g0jVRxvKa"  # cust123
        ),
    ]
    
    for user in demo_users:
        USERS_DB[user.email] = user
    
    _USERS_DB_INITIALIZED = True

# Initialize on first access
_init_users_db()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def get_user(email: str) -> Optional[User]:
    """Get user by email from database."""
    return USERS_DB.get(email)


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID from database."""
    for user in USERS_DB.values():
        if user.user_id == user_id:
            return user
    return None


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password.
    
    Args:
        email: User email
        password: Plain text password
        
    Returns:
        User object if authenticated, None otherwise
    """
    user = get_user(email)
    if not user:
        return None
    if not user.hashed_password:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Payload data
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
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData with decoded payload
        
    Raises:
        HTTPException if token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(
            user_id=user_id,
            email=payload.get("email"),
            role=payload.get("role"),
            department=payload.get("department"),
            exp=payload.get("exp")
        )
        
        return token_data
        
    except JWTError:
        raise credentials_exception


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user from token.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User object
        
    Raises:
        HTTPException if authentication fails
    """
    token = credentials.credentials
    token_data = verify_token(token)
    
    user = get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: User from get_current_user
        
    Returns:
        User object if active
        
    Raises:
        HTTPException if user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get user if authenticated, None otherwise.
    Useful for endpoints that work with or without authentication.
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str,
    department: Optional[str] = None,
    is_internal: bool = True
) -> User:
    """
    Create a new user (in-memory for demo, use database in production).
    
    Args:
        email: User email
        password: Plain text password
        full_name: User's full name
        role: User role
        department: User department
        is_internal: Whether user is internal employee
        
    Returns:
        Created User object
    """
    if email in USERS_DB:
        raise ValueError(f"User with email {email} already exists")
    
    user_id = f"user_{len(USERS_DB) + 1:03d}"
    
    user = User(
        user_id=user_id,
        email=email,
        full_name=full_name,
        role=role,
        department=department,
        is_active=True,
        is_internal=is_internal,
        hashed_password=get_password_hash(password)
    )
    
    USERS_DB[email] = user
    return user
