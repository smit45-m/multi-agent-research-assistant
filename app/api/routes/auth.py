"""Authentication and authorization endpoints (JWT signup, login, session revocation, password management)."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth.schemas import (
    AdminUserUpdateRequest,
    ChangePasswordRequest,
    SessionInfo,
    TokenResponse,
    UserLoginRequest,
    UserResponse,
    UserSignUpRequest,
)
from app.auth.database import get_auth_db, AuthDatabase
from app.auth.security import (
    create_access_token,
    get_current_user,
    require_admin,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def _to_user_response(user: dict) -> UserResponse:
    return UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        db_name=user.get("db_name", ""),
        is_active=bool(user.get("is_active", True)),
        created_at=user["created_at"],
        last_login=user.get("last_login"),
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    req: UserSignUpRequest,
    request: Request,
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """
    Registers a new researcher account, provisions a separate isolated MongoDB database
    for this user, and issues a stateful Bearer JWT access token.
    Public registration cannot self-assign admin privileges.
    """
    if req.role and req.role.lower() != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration cannot assign administrator privileges.",
        )

    try:
        user = auth_db.create_user(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            role="user",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    user_agent = request.headers.get("user-agent", "")
    token, expires_in = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        full_name=user["full_name"],
        db_name=user["db_name"],
        user_agent=user_agent,
    )

    auth_db.record_activity(user["id"], "signup", f"Provisioned private DB '{user['db_name']}' for {user['email']}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=_to_user_response(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    req: UserLoginRequest,
    request: Request,
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Authenticates a user via email & password with brute-force lockout protection and stateful session issuance."""
    user = auth_db.get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    retry_after = auth_db.is_login_locked(user)
    if retry_after:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
        )

    if not auth_db.verify_password(req.password, user["password_hash"]):
        auth_db.record_failed_login(user["id"])
        auth_db.record_activity(user["id"], "failed_login", "Invalid password attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Please contact an administrator.",
        )

    db_name = auth_db.ensure_user_db(user)
    auth_db.update_last_login(user["id"])
    auth_db.record_activity(user["id"], "login", f"Authenticated into private DB '{db_name}'")

    user_agent = request.headers.get("user-agent", "")
    token, expires_in = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        full_name=user["full_name"],
        db_name=db_name,
        user_agent=user_agent,
    )

    refreshed = auth_db.get_user_by_id(user["id"]) or user
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=_to_user_response(refreshed),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserResponse = Depends(get_current_user),
):
    """Returns the currently authenticated user's profile, role, and private MongoDB database ID."""
    return current_user


@router.post("/logout")
async def logout(
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Revokes the active JWT session (`jti`) server-side so the token cannot be reused."""
    jti = getattr(request.state, "auth_jti", "")
    if jti:
        auth_db.revoke_session(jti, user_id=current_user.id)
    auth_db.record_activity(current_user.id, "logout", f"Session {jti[:8]} revoked for {current_user.email}")
    return {"message": "Successfully logged out and revoked session token.", "status": "logged_out"}


@router.post("/logout-all")
async def logout_all_sessions(
    current_user: UserResponse = Depends(get_current_user),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Revokes all active sessions across devices for the authenticated user."""
    count = auth_db.revoke_all_sessions(current_user.id)
    auth_db.record_activity(current_user.id, "logout_all", f"Revoked {count} active sessions")
    return {"message": f"Revoked {count} active session(s).", "revoked_count": count}


@router.get("/sessions", response_model=List[SessionInfo])
async def list_my_sessions(
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Lists active and recent login sessions strictly for the authenticated user."""
    current_jti = getattr(request.state, "auth_jti", "")
    rows = auth_db.list_user_sessions(current_user.id)
    return [
        SessionInfo(
            jti=s["jti"],
            user_agent=s.get("user_agent", ""),
            created_at=s.get("created_at", ""),
            expires_at=s.get("expires_at", ""),
            revoked=bool(s.get("revoked", False)),
            is_current=(s["jti"] == current_jti),
        )
        for s in rows
    ]


@router.post("/change-password", response_model=TokenResponse)
async def change_password(
    req: ChangePasswordRequest,
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Changes user password, revokes prior sessions, and returns a fresh session token."""
    user = auth_db.get_user_by_id(current_user.id)
    if not user or not auth_db.verify_password(req.current_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    auth_db.update_password(current_user.id, req.new_password)
    auth_db.record_activity(current_user.id, "change_password", "Password updated & prior sessions revoked")

    token, expires_in = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        full_name=user["full_name"],
        db_name=current_user.db_name,
        user_agent=request.headers.get("user-agent", ""),
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=current_user,
    )


@router.get("/users", response_model=List[UserResponse])
async def list_registered_users(
    admin: UserResponse = Depends(require_admin),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Admin-only: returns registered account metadata (never private per-user chats/files)."""
    return [_to_user_response(u) for u in auth_db.list_users()]


@router.patch("/users/{user_id}", response_model=UserResponse)
async def admin_update_user(
    user_id: str,
    req: AdminUserUpdateRequest,
    admin: UserResponse = Depends(require_admin),
    auth_db: AuthDatabase = Depends(get_auth_db),
):
    """Admin-only: promote/demote role or activate/deactivate an account."""
    updated = auth_db.update_user_admin_fields(user_id, role=req.role, is_active=req.is_active)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    auth_db.record_activity(admin.id, "admin_update_user", f"Updated user {user_id}: role={req.role}, active={req.is_active}")
    return _to_user_response(updated)
