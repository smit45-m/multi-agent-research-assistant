"""JWT token creation, validation, and FastAPI authorization dependencies."""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.auth.database import get_auth_db, AuthDatabase
from app.auth.schemas import UserResponse

security_scheme = HTTPBearer(auto_error=False)
mandatory_security_scheme = HTTPBearer(auto_error=True)


def create_access_token(
    user_id: str,
    email: str,
    role: str = "user",
    full_name: str = "",
    expires_delta: Optional[timedelta] = None
) -> Tuple[str, int]:
    """Generates a signed JWT access token with payload and expiration."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
        expires_seconds = int(expires_delta.total_seconds())
    else:
        expires_seconds = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        expire = now + timedelta(seconds=expires_seconds)

    payload: Dict[str, Any] = {
        "sub": user_id,
        "email": email.strip().lower(),
        "role": role,
        "name": full_name,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access_token"
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return token, expires_seconds


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates JWT token signature and expiration."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(mandatory_security_scheme),
    auth_db: AuthDatabase = Depends(get_auth_db)
) -> UserResponse:
    """Enforces authentication: verifies JWT token and active user record."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated."
        )

    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
        last_login=user.get("last_login")
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    auth_db: AuthDatabase = Depends(get_auth_db)
) -> Optional[UserResponse]:
    """Allows optional authentication: returns user profile if valid token provided, else None."""
    if not credentials or not credentials.credentials:
        return None

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = auth_db.get_user_by_id(user_id)
        if not user or not user.get("is_active", True):
            return None
        return UserResponse(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            is_active=bool(user["is_active"]),
            created_at=user["created_at"],
            last_login=user.get("last_login")
        )
    except Exception:
        return None


async def require_admin(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Enforces Role-Based Access Control (RBAC): user must possess admin role."""
    if current_user.role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this resource."
        )
    return current_user
