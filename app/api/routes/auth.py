"""Authentication and authorization endpoints (JWT signup, login, session inspection, and user management)."""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.schemas import (
    UserSignUpRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse
)
from app.auth.database import get_auth_db, AuthDatabase
from app.auth.security import (
    create_access_token,
    get_current_user,
    require_admin
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    req: UserSignUpRequest,
    auth_db: AuthDatabase = Depends(get_auth_db)
):
    """
    Registers a new user account with email and password, returning a signed JWT access token.
    """
    try:
        user = auth_db.create_user(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            role=req.role or "user"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Generate JWT access token
    token, expires_in = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        full_name=user["full_name"]
    )

    auth_db.record_activity(user["id"], "signup", f"Account created for {user['email']}")

    user_resp = UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
        last_login=user.get("last_login")
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=user_resp
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    req: UserLoginRequest,
    auth_db: AuthDatabase = Depends(get_auth_db)
):
    """
    Authenticates a user via email and password, returning a signed JWT access token upon success.
    """
    user = auth_db.get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not auth_db.verify_password(req.password, user["password_hash"]):
        auth_db.record_activity(user["id"], "failed_login", "Invalid password attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Please contact an administrator."
        )

    # Update last login timestamp
    auth_db.update_last_login(user["id"])
    auth_db.record_activity(user["id"], "login", "Successful authentication")

    # Generate JWT token
    token, expires_in = create_access_token(
        user_id=user["id"],
        email=user["email"],
        role=user["role"],
        full_name=user["full_name"]
    )

    user_resp = UserResponse(
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
        last_login=user.get("last_login")
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=user_resp
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Returns the currently authenticated user's profile and role.
    Requires Bearer token in Authorization header.
    """
    return current_user


@router.post("/logout")
async def logout(
    current_user: UserResponse = Depends(get_current_user),
    auth_db: AuthDatabase = Depends(get_auth_db)
):
    """
    Terminates the user's active session and logs audit event.
    Client should discard stored JWT token upon calling this endpoint.
    """
    auth_db.record_activity(current_user.id, "logout", f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out.", "status": "logged_out"}


@router.get("/users", response_model=List[UserResponse])
async def list_registered_users(
    admin: UserResponse = Depends(require_admin),
    auth_db: AuthDatabase = Depends(get_auth_db)
):
    """
    Admin-only endpoint: returns all registered user accounts and their assigned roles.
    """
    raw_users = auth_db.list_users()
    return [
        UserResponse(
            id=u["id"],
            email=u["email"],
            full_name=u["full_name"],
            role=u["role"],
            is_active=bool(u["is_active"]),
            created_at=u["created_at"],
            last_login=u.get("last_login")
        )
        for u in raw_users
    ]
