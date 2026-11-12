"""Pydantic schemas for User Authentication, Authorization, and Session Management."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class UserSignUpRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Valid user email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password (min 6 characters)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name or display name")
    role: Optional[str] = Field(default="user", description="Account role: public registration only permits 'user'")


class UserLoginRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Registered email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    db_name: str = ""
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds
    user: UserResponse


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)


class SessionInfo(BaseModel):
    jti: str
    user_agent: str = ""
    created_at: str
    expires_at: str
    revoked: bool = False
    is_current: bool = False


class AdminUserUpdateRequest(BaseModel):
    role: Optional[Literal["user", "admin"]] = None
    is_active: Optional[bool] = None
