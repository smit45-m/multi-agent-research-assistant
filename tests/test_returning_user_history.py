"""Tests confirming FAISS serves vector tasks while MongoDB persists past queries/outputs across logins."""
from __future__ import annotations

import uuid

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db.mongo_engine import reset_mongo_manager
from app.db.user_repository import UserWorkspaceRepository, _USER_VECTOR_STORES, get_user_vector_store
from app.main import app
from app.rag.vector_store import VectorStoreManager


@pytest.fixture
def returning_mongo_env(tmp_path, monkeypatch):
    workspaces_dir = tmp_path / "user_workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    monkeypatch.setattr(settings, "USER_WORKSPACES_DIR", str(workspaces_dir))
    _USER_VECTOR_STORES.clear()

    mock_client = mongomock.MongoClient()
    mgr = reset_mongo_manager(client=mock_client)
    try:
        with TestClient(app) as client:
            yield app, client, mgr, workspaces_dir
    finally:
        import app.db.mongo_engine as mongo_engine_mod

        _USER_VECTOR_STORES.clear()
        mongo_engine_mod._MANAGER_INSTANCE = None


def test_returning_user_accesses_past_queries_and_full_outputs_across_logins(returning_mongo_env):
    """Simulates Visit 1 -> logout -> Visit 2 login and verifies MongoDB query/output retrieval + search."""
    _app, client, mgr, _workspaces = returning_mongo_env

    # Visit 1: user signs up
    signup = client.post(
        "/api/v1/auth/signup",
        json={
            "email": "returning.user@example.com",
            "password": "Password123!",
            "full_name": "Returning Researcher",
        },
    )
    assert signup.status_code == 201, signup.text
    token1 = signup.json()["access_token"]
    user = signup.json()["user"]

    # Record a long output where a searchable phrase only appears deep in the full report (past 380-char summary)
    filler = "Background overview sentence. " * 25
    deep_detail = "CRYOGENIC_CALIBRATION_TOKEN_42K telemetry confirmed."
    full_report = f"{filler}\n\n## Deep Measurements\n{deep_detail}"
    thread_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())

    repo = UserWorkspaceRepository(user["db_name"], user["id"])
    repo.record_chat_turn(
        chat_id=chat_id,
        thread_id=thread_id,
        query="What were the dilution refrigerator sensor readings?",
        report=full_report,
        sources=[
            {
                "title": "cryo_notes.pdf",
                "url_or_path": "cryo_notes.pdf",
                "source_type": "document",
                "snippet": deep_detail,
            }
        ],
        mode="fast",
        rag_mode="hybrid",
        orchestrator="direct",
        model="gemini-flash-lite-latest",
        answer_origin="document",
        confidence_score=0.95,
        response_accuracy_score=0.94,
        processing_time_seconds=1.1,
    )

    # Confirm user logs out cleanly at end of Visit 1
    logout = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token1}"})
    assert logout.status_code == 200
    assert client.get("/api/v1/history/threads", headers={"Authorization": f"Bearer {token1}"}).status_code == 401

    # Simulate process/session restart using the same underlying persistent mock client
    new_mgr = reset_mongo_manager(client=mgr.client)
    assert new_mgr.client is mgr.client

    # Visit 2: returning user signs in a second time
    login2 = client.post(
        "/api/v1/auth/login",
        json={"email": "returning.user@example.com", "password": "Password123!"},
    )
    assert login2.status_code == 200, login2.text
    token2 = login2.json()["access_token"]
    headers2 = {"Authorization": f"Bearer {token2}"}

    # Verify FAISS still handles per-user vector store cleanly
    assert isinstance(get_user_vector_store(user["db_name"]), VectorStoreManager)

    # Search threads by phrase present ONLY deep in the full output
    threads_res = client.get(
        "/api/v1/history/threads",
        params={"search": "CRYOGENIC_CALIBRATION_TOKEN_42K"},
        headers=headers2,
    )
    assert threads_res.status_code == 200
    threads_data = threads_res.json()
    assert threads_data["total_threads"] == 1
    turn = threads_data["threads"][0]["turns"][0]
    assert turn["query"] == "What were the dilution refrigerator sensor readings?"
    assert deep_detail in turn["report"]

    # Search individual chats by phrase present ONLY deep in the full output
    chats_res = client.get(
        "/api/v1/history/chats",
        params={"search": "CRYOGENIC_CALIBRATION_TOKEN_42K"},
        headers=headers2,
    )
    assert chats_res.status_code == 200
    assert chats_res.json()["total_chats"] == 1
    assert chats_res.json()["chats"][0]["chat_id"] == chat_id

    # Fetch exact single chat record on Visit 2
    single_res = client.get(f"/api/v1/history/chats/{chat_id}", headers=headers2)
    assert single_res.status_code == 200
    assert deep_detail in single_res.json()["report"]
