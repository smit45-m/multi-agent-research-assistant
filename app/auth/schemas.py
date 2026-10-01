"""Pydantic schemas for User Authentication and Authorization."""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserSignUpRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Valid user email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password (min 6 characters)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name or display name")
    role: Optional[str] = Field(default="user", description="Account role: 'user' or 'admin'")


class UserLoginRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Registered email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds
    user: UserResponse


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(None, min_length=6, max_length=128)
