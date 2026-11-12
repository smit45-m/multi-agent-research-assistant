"""JWT token creation, stateful session validation, and FastAPI authorization dependencies."""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import uuid
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.auth.database import get_auth_db, AuthDatabase
from app.auth.schemas import UserResponse

security_scheme = HTTPBearer(auto_error=False)
mandatory_security_scheme = HTTPBearer(auto_error=False)


def create_access_token(
    user_id: str,
    email: str,
    role: str = "user",
    full_name: str = "",
    db_name: str = "",
    user_agent: str = "",
    expires_delta: Optional[timedelta] = None,
    auth_db: Optional[AuthDatabase] = None,
) -> Tuple[str, int]:
    """Generates a signed JWT access token bound to a persistent `jti` session and user DB."""
    settings = get_settings()
    auth_db = auth_db or get_auth_db()
    if not db_name:
        user_rec = auth_db.get_user_by_id(user_id)
        if user_rec:
            db_name = auth_db.ensure_user_db(user_rec)

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
        expires_seconds = int(expires_delta.total_seconds())
    else:
        expires_seconds = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        expire = now + timedelta(seconds=expires_seconds)

    jti = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        "sub": user_id,
        "jti": jti,
        "email": email.strip().lower(),
        "role": role,
        "name": full_name,
        "db_name": db_name,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access_token",
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    auth_db.create_session(
        jti=jti,
        user_id=user_id,
        db_name=db_name,
        expires_at=expire.isoformat(),
        user_agent=user_agent,
    )
    return token, expires_seconds


def decode_access_token(
    token: str,
    *,
    verify_session: bool = True,
    auth_db: Optional[AuthDatabase] = None,
) -> Dict[str, Any]:
    """Decodes and validates JWT signature, expiration, token type, and stateful session status."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access_token" or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed authentication token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if verify_session:
        jti = payload.get("jti", "")
        session_db = auth_db or get_auth_db()
        if not jti or not session_db.is_session_active(jti, payload["sub"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has been signed out or revoked. Please sign in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return payload


def _resolve_user_from_token(token: str, auth_db: AuthDatabase) -> Tuple[UserResponse, Dict[str, Any]]:
    payload = decode_access_token(token, verify_session=True)
    user_id = payload["sub"]
    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    db_name = auth_db.ensure_user_db(user)
    token_db = payload.get("db_name")
    if token_db and token_db != db_name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token database binding mismatch. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_resp = UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        db_name=db_name,
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
        last_login=user.get("last_login"),
    )
    return user_resp, payload


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(mandatory_security_scheme),
    auth_db: AuthDatabase = Depends(get_auth_db),
) -> UserResponse:
    """Enforces strict authentication: verifies valid, non-revoked JWT session and active user."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in with a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_resp, payload = _resolve_user_from_token(credentials.credentials, auth_db)
    request.state.auth_jti = payload.get("jti", "")
    request.state.current_user = user_resp
    return user_resp


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    auth_db: AuthDatabase = Depends(get_auth_db),
) -> Optional[UserResponse]:
    """
    Faithful optional auth:
    - If NO Authorization header was sent -> returns None (unauthenticated guest sandbox).
    - If an Authorization header WAS sent -> strictly validates it; invalid/expired/revoked tokens raise 401!
    """
    auth_header = request.headers.get("Authorization")
    if not credentials and not auth_header:
        return None
    if auth_header and (not credentials or not credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_resp, payload = _resolve_user_from_token(credentials.credentials, auth_db)
    request.state.auth_jti = payload.get("jti", "")
    request.state.current_user = user_resp
    return user_resp


async def require_admin(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """Enforces Role-Based Access Control (RBAC): user must possess admin role."""
    if current_user.role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to access this resource.",
        )
    return current_user
