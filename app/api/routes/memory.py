"""Per-user private short-term & long-term memory inspection and management endpoints."""
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth.schemas import UserResponse
from app.auth.security import get_current_user
from app.db.user_memory import UserMemoryService
from app.db.user_repository import UserWorkspaceRepository

router = APIRouter(prefix="/api/v1/memory", tags=["User Memory"])


class PinMemoryRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=140)
    summary: str = Field(..., min_length=3, max_length=800)
    kind: Literal["pinned_note", "user_preference"] = "pinned_note"


@router.get("/")
async def get_my_memory(
    current_user: UserResponse = Depends(get_current_user),
):
    """Inspects short-term & long-term memory strictly inside the signed-in user's MongoDB DB."""
    mem = UserMemoryService(current_user.db_name, current_user.id)
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    data = mem.list_memories()
    return {
        "user_id": current_user.id,
        "db_name": current_user.db_name,
        "stats": repo.get_workspace_stats(),
        **data,
    }


@router.post("/long-term", status_code=status.HTTP_201_CREATED)
async def add_long_term_memory(
    req: PinMemoryRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Pins a persistent preference or long-term memory item strictly for the signed-in user."""
    mem = UserMemoryService(current_user.db_name, current_user.id)
    created = mem.add_pinned_memory(req.title, req.summary, kind=req.kind)
    return created


@router.delete("/long-term/{memory_id}")
async def delete_long_term_memory(
    memory_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Deletes a specific long-term memory strictly from the signed-in user's database."""
    mem = UserMemoryService(current_user.db_name, current_user.id)
    if not mem.delete_long_term(memory_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory item not found in your database.")
    return {"status": "deleted", "memory_id": memory_id}


@router.delete("/short-term")
async def clear_short_term_memory(
    current_user: UserResponse = Depends(get_current_user),
):
    """Clears active short-term conversation buffer strictly for the signed-in user."""
    mem = UserMemoryService(current_user.db_name, current_user.id)
    cleared = mem.clear_short_term()
    return {"status": "cleared", "cleared_count": cleared}
