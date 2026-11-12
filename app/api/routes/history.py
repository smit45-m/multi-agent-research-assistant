"""Per-user private chat history, thread retrieval, and file-to-Q&A provenance routes."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth.schemas import UserResponse
from app.auth.security import get_current_user
from app.db.user_repository import UserWorkspaceRepository

router = APIRouter(prefix="/api/v1/history", tags=["User History"])


@router.get("/threads")
async def list_my_threads(
    search: Optional[str] = Query(default=None),
    limit: int = Query(default=40, ge=1, le=100),
    current_user: UserResponse = Depends(get_current_user),
):
    """Returns all private conversation threads and Q&A turns strictly for the signed-in user."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    threads = repo.list_threads(search=search, limit=limit)
    return {
        "user_id": current_user.id,
        "db_name": current_user.db_name,
        "threads": threads,
        "total_threads": len(threads),
        "stats": repo.get_workspace_stats(),
    }


@router.get("/threads/{thread_id}")
async def get_my_thread(
    thread_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Returns full ordered turns of a specific thread belonging strictly to the signed-in user."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    thread = repo.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation thread not found in your workspace.")
    return thread


@router.delete("/threads/{thread_id}")
async def delete_my_thread(
    thread_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Deletes an entire conversation thread strictly from the signed-in user's database."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    deleted = repo.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation thread not found in your workspace.")
    return {"status": "deleted", "thread_id": thread_id, "deleted_turns": deleted}


@router.get("/chats")
async def list_my_chats(
    thread_id: Optional[str] = Query(default=None),
    document_id: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: UserResponse = Depends(get_current_user),
):
    """Returns individual past questions & answers strictly from the signed-in user's MongoDB DB."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    chats = repo.list_chats(thread_id=thread_id, document_id=document_id, search=search, limit=limit)
    return {
        "user_id": current_user.id,
        "db_name": current_user.db_name,
        "chats": chats,
        "total_chats": len(chats),
    }


@router.get("/chats/{chat_id}")
async def get_my_chat(
    chat_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Retrieves a specific past Q&A record strictly from the signed-in user's database."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    chat = repo.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat record not found in your workspace.")
    return chat


@router.delete("/chats/{chat_id}")
async def delete_my_chat(
    chat_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Deletes a single Q&A record strictly from the signed-in user's database."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    if not repo.delete_chat(chat_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat record not found in your workspace.")
    return {"status": "deleted", "chat_id": chat_id}


@router.get("/files")
async def list_my_files_with_qa(
    current_user: UserResponse = Depends(get_current_user),
):
    """Lists files uploaded by the signed-in user alongside every Question & Answer generated from each file."""
    repo = UserWorkspaceRepository(current_user.db_name, current_user.id)
    files = repo.list_files()
    return {
        "user_id": current_user.id,
        "db_name": current_user.db_name,
        "files": files,
        "total_files": len(files),
    }
