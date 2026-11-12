"""End-to-end integration tests for per-user MongoDB database isolation,
short-term & long-term memory persistence, file-to-Q&A provenance, and faithful auth.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import bcrypt
import pytest
from fastapi.testclient import TestClient

from app.agents.graph import ResearchGraph
from app.api.routes import documents as documents_routes
from app.api.routes import research as research_routes
from app.auth.database import AuthDatabase
from app.config import get_settings
from app.db.mongo_engine import get_mongo_manager, reset_mongo_manager
from app.db.user_memory import UserMemoryService
from app.db.user_repository import UserWorkspaceRepository, clear_cached_user_vector_stores
from app.main import create_app


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provides completely isolated embedded MongoDB directories, user vector stores, and FastAPI app."""
    mongo_dir = tmp_path / "mongo_dbs"
    workspaces_dir = tmp_path / "user_workspaces"
    legacy_auth = tmp_path / "no_legacy_auth.db"

    settings = get_settings()
    monkeypatch.setattr(settings, "MONGODB_URI", None, raising=False)
    monkeypatch.setattr(settings, "MONGODB_REQUIRE_OFFICIAL", False, raising=False)
    monkeypatch.setattr(settings, "MONGODB_DATA_DIR", str(mongo_dir), raising=False)
    monkeypatch.setattr(settings, "USER_WORKSPACES_DIR", str(workspaces_dir), raising=False)
    monkeypatch.setattr(settings, "AUTH_DB_PATH", str(legacy_auth), raising=False)

    clear_cached_user_vector_stores()
    documents_routes.DOCUMENTS.clear()
    research_routes.TASKS.clear()
    research_routes.TASK_OWNERS.clear()
    with research_routes._SYNTHESIS_CACHE_LOCK:
        research_routes._SYNTHESIS_CACHE.clear()

    reset_mongo_manager(mongodb_uri="", data_dir=mongo_dir, require_official=False)
    app = create_app()
    with TestClient(app) as client:
        yield {
            "client": client,
            "mongo_dir": mongo_dir,
            "workspaces_dir": workspaces_dir,
            "tmp_path": tmp_path,
        }

    clear_cached_user_vector_stores()
    documents_routes.DOCUMENTS.clear()
    research_routes.TASKS.clear()
    research_routes.TASK_OWNERS.clear()


def _signup(client: TestClient, email: str, password: str, full_name: str, role: str = "user"):
    return client.post(
        "/api/v1/auth/signup",
        json={"email": email, "password": password, "full_name": full_name, "role": role},
    )


def _login(client: TestClient, email: str, password: str):
    return client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )


def test_separate_per_user_databases_and_vector_stores(isolated_env):
    client: TestClient = isolated_env["client"]
    mongo_dir: Path = isolated_env["mongo_dir"]
    workspaces_dir: Path = isolated_env["workspaces_dir"]

    r_alice = _signup(client, "alice@example.com", "AlicePass#1", "Alice Researcher")
    r_bob = _signup(client, "bob@example.com", "BobPass#1", "Bob Analyst")

    assert r_alice.status_code == 201
    assert r_bob.status_code == 201

    alice = r_alice.json()
    bob = r_bob.json()
    alice_db = alice["user"]["db_name"]
    bob_db = bob["user"]["db_name"]

    assert alice_db.startswith("scholar_u_")
    assert bob_db.startswith("scholar_u_")
    assert alice_db != bob_db
    assert (mongo_dir / alice_db).is_dir()
    assert (mongo_dir / bob_db).is_dir()

    mgr = get_mongo_manager()
    user_dbs = mgr.list_user_databases()
    assert alice_db in user_dbs
    assert bob_db in user_dbs
    assert mgr.auth_db_name not in user_dbs

    # Uploading a file for Alice initializes ONLY Alice's isolated vector store directory
    up_res = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {alice['access_token']}"},
        files={"file": ("alice_notes.txt", b"Private lattice cryptography notes for Alice.", "text/plain")},
    )
    assert up_res.status_code == 200
    assert (workspaces_dir / alice_db / "vector_store").exists()
    assert not (workspaces_dir / bob_db / "vector_store").exists()


def test_faithful_auth_privilege_escalation_rbac_and_lockout(isolated_env):
    client: TestClient = isolated_env["client"]

    # Self-service admin signup must be strictly rejected
    escalate = _signup(client, "attacker@example.com", "Attack#123", "Attacker", role="admin")
    assert escalate.status_code == 403

    # Normal researcher cannot hit admin routes
    user_res = _signup(client, "normal@example.com", "Normal#123", "Normal User")
    assert user_res.status_code == 201
    user_token = user_res.json()["access_token"]
    assert client.get("/api/v1/auth/users", headers={"Authorization": f"Bearer {user_token}"}).status_code == 403

    # Seeded default admin can log in and inspect users
    admin_res = _login(client, "admin@research.com", "Admin@1234")
    assert admin_res.status_code == 200
    assert admin_res.json()["user"]["role"] == "admin"
    users_list = client.get(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {admin_res.json()['access_token']}"},
    )
    assert users_list.status_code == 200
    assert any(u["email"] == "normal@example.com" for u in users_list.json())

    # Brute-force lockout after 6 wrong passwords
    for _ in range(6):
        bad = _login(client, "normal@example.com", "WrongPassword")
        assert bad.status_code == 401
    locked = _login(client, "normal@example.com", "Normal#123")
    assert locked.status_code == 429


def test_stateful_jwt_revocation_and_strict_bearer_rejection(isolated_env):
    client: TestClient = isolated_env["client"]

    # Protected endpoints require valid Bearer auth
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/history/threads").status_code == 401
    assert client.get("/api/v1/memory/").status_code == 401

    # Invalid Bearer tokens must fail with 401 even on optional-auth routes (never degrade to guest)
    bad_headers = {"Authorization": "Bearer forged.jwt.token"}
    assert client.get("/api/v1/documents/", headers=bad_headers).status_code == 401
    assert client.post("/api/v1/research/sync", headers=bad_headers, json={"query": "hello world"}).status_code == 401

    # Guest access without Authorization header remains allowed on sandbox documents endpoint
    assert client.get("/api/v1/documents/").status_code == 200

    s1 = _signup(client, "carol@example.com", "CarolPass#1", "Carol").json()
    t1 = s1["access_token"]
    s2 = _login(client, "carol@example.com", "CarolPass#1").json()
    t2 = s2["access_token"]

    sessions_res = client.get("/api/v1/auth/sessions", headers={"Authorization": f"Bearer {t1}"})
    assert sessions_res.status_code == 200
    assert len(sessions_res.json()) == 2
    assert sum(1 for s in sessions_res.json() if s["is_current"]) == 1

    # Logging out t1 revokes ONLY t1 immediately on server
    out_res = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {t1}"})
    assert out_res.status_code == 200
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t1}"}).status_code == 401
    assert client.get("/api/v1/history/threads", headers={"Authorization": f"Bearer {t1}"}).status_code == 401
    assert client.get("/api/v1/documents/", headers={"Authorization": f"Bearer {t1}"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).status_code == 200

    # Password change issues a fresh token and revokes t2
    cp_res = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {t2}"},
        json={"current_password": "CarolPass#1", "new_password": "CarolNewPass#2"},
    )
    assert cp_res.status_code == 200
    t3 = cp_res.json()["access_token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t3}"}).status_code == 200

    # logout-all revokes t3 too
    assert client.post("/api/v1/auth/logout-all", headers={"Authorization": f"Bearer {t3}"}).status_code == 200
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t3}"}).status_code == 401


def test_isolation_file_qa_links_and_short_long_term_memory_across_restart(isolated_env, monkeypatch):
    client: TestClient = isolated_env["client"]
    mongo_dir: Path = isolated_env["mongo_dir"]

    alice = _signup(client, "alice2@example.com", "AlicePass#1", "Alice Researcher").json()
    bob = _signup(client, "bob2@example.com", "BobPass#1", "Bob Analyst").json()
    alice_hdr = {"Authorization": f"Bearer {alice['access_token']}"}
    bob_hdr = {"Authorization": f"Bearer {bob['access_token']}"}

    # Alice uploads her private document
    up = client.post(
        "/api/v1/documents/upload",
        headers=alice_hdr,
        files={
            "file": (
                "alice_secret_spec.txt",
                b"Quantum Lattice Key Agreement Protocol uses 4096-bit ring signatures and rotates keys every 15 minutes.",
                "text/plain",
            )
        },
    )
    assert up.status_code == 200
    alice_doc_id = up.json()["document_id"]

    recorded_graph_calls: List[Dict[str, Any]] = []

    def fake_run(self, query: str, **kwargs):
        recorded_graph_calls.append(
            {
                "query": query,
                "user_vector_store": kwargs.get("user_vector_store"),
                "user_memory_context": kwargs.get("user_memory_context", ""),
                "user_memory_meta": kwargs.get("user_memory_meta", {}),
                "attachment_ids": kwargs.get("attachment_ids"),
            }
        )
        return {
            "final_report": (
                "### Synthesis\nBased on `alice_secret_spec.txt`, keys rotate every **15 minutes** "
                "using 4096-bit ring signatures."
            ),
            "sources_cited": [
                {
                    "title": "alice_secret_spec.txt",
                    "url_or_path": f"attachment:{alice_doc_id}",
                    "snippet": "rotates keys every 15 minutes.",
                    "source_type": "txt",
                    "relevance_score": 0.95,
                }
            ],
            "agent_telemetry": {"total_time_ms": 120.0},
            "confidence_score": 0.94,
            "response_accuracy_score": 0.92,
            "synthesis_speedup_ratio": 0.65,
            "orchestrator": "langgraph",
            "model": "gemini-2.5-flash",
            "answer_origin": "cloud_synthesized",
        }

    monkeypatch.setattr(ResearchGraph, "run", fake_run)

    # Turn 1 for Alice: states preference + cites her private doc
    q1 = "Explain my Quantum Lattice Protocol schedule and remember my preference: answer in concise bullet tables."
    res1 = client.post(
        "/api/v1/research/sync",
        headers=alice_hdr,
        json={"query": q1, "mode": "research", "orchestrator": "langgraph", "attachment_ids": [alice_doc_id]},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    thread_id = d1["thread_id"]
    task1_id = d1["task_id"]
    assert thread_id
    assert len(recorded_graph_calls) == 1
    assert recorded_graph_calls[0]["user_vector_store"] is not None

    # Turn 2 follow-up for Alice in the same thread: verify short & long-term memory injected!
    q2 = "Can you clarify that rotation interval?"
    res2 = client.post(
        "/api/v1/research/sync",
        headers=alice_hdr,
        json={"query": q2, "mode": "research", "orchestrator": "langgraph", "thread_id": thread_id},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["thread_id"] == thread_id
    assert d2["memory_context_used"]["short_term_count"] >= 1
    assert d2["memory_context_used"]["long_term_count"] >= 1
    injected = recorded_graph_calls[1]["user_memory_context"]
    assert "Short-Term Conversation Context" in injected
    assert "Long-Term Remembered Context" in injected
    assert "Quantum Lattice Protocol" in injected

    # File-to-Q&A linking visible strictly to Alice
    alice_files = client.get("/api/v1/history/files", headers=alice_hdr).json()
    assert alice_files["total_files"] == 1
    assert alice_files["files"][0]["qa_count"] == 2
    assert alice_files["files"][0]["qa_links"][0]["chat_id"] == d2["chat_id"]

    # Bob must see ZERO files, ZERO chats, ZERO memories, 404 deleting Alice's file, 403 accessing Alice's task
    assert client.get("/api/v1/documents/", headers=bob_hdr).json()["total_count"] == 0
    assert client.get("/api/v1/history/files", headers=bob_hdr).json()["total_files"] == 0
    assert client.get("/api/v1/history/threads", headers=bob_hdr).json()["total_threads"] == 0
    bob_mem = client.get("/api/v1/memory/", headers=bob_hdr).json()
    assert bob_mem["short_term_count"] == 0
    assert bob_mem["long_term_count"] == 0
    assert client.delete(f"/api/v1/documents/{alice_doc_id}", headers=bob_hdr).status_code == 404
    assert client.get(f"/api/v1/research/{task1_id}", headers=bob_hdr).status_code == 403

    # Bob's own research run receives NO Alice memory and NO Alice vector store docs
    res_bob = client.post(
        "/api/v1/research/sync",
        headers=bob_hdr,
        json={"query": "What is quantum lattice?", "mode": "research", "orchestrator": "langgraph"},
    )
    assert res_bob.status_code == 200
    assert recorded_graph_calls[2]["user_memory_context"] == ""
    assert len(recorded_graph_calls[2]["user_vector_store"]._documents) == 0

    # Simulate complete server restart: wipe in-memory caches and reload Mongo manager from disk
    clear_cached_user_vector_stores()
    research_routes.TASKS.clear()
    research_routes.TASK_OWNERS.clear()
    reset_mongo_manager(mongodb_uri="", data_dir=mongo_dir, require_official=False)
    fresh_app = create_app()
    with TestClient(fresh_app) as client2:
        alice_login = _login(client2, "alice2@example.com", "AlicePass#1")
        assert alice_login.status_code == 200
        a2_hdr = {"Authorization": f"Bearer {alice_login.json()['access_token']}"}

        threads_after = client2.get("/api/v1/history/threads", headers=a2_hdr).json()
        assert threads_after["total_threads"] == 1
        assert threads_after["threads"][0]["turn_count"] == 2

        files_after = client2.get("/api/v1/history/files", headers=a2_hdr).json()
        assert files_after["total_files"] == 1
        assert files_after["files"][0]["qa_count"] == 2

        mem_after = client2.get("/api/v1/memory/", headers=a2_hdr).json()
        assert mem_after["short_term_count"] == 2
        assert mem_after["long_term_count"] >= 1

        recalled_after = UserMemoryService(alice["user"]["db_name"], alice["user"]["id"]).recall_context(
            "Tell me more about that schedule",
            thread_id=thread_id,
        )
        assert "Short-Term Conversation Context" in recalled_after["formatted_prompt_block"]
        assert "Long-Term Remembered Context" in recalled_after["formatted_prompt_block"]


def test_streaming_endpoint_persists_chats_and_memory(isolated_env, monkeypatch):
    client: TestClient = isolated_env["client"]
    dave = _signup(client, "dave@example.com", "DavePass#1", "Dave").json()
    dave_hdr = {"Authorization": f"Bearer {dave['access_token']}"}

    def fake_stream_run(self, query: str, on_token=None, on_stage=None, **kwargs):
        if on_stage:
            on_stage("write", "Writing report")
        if on_token:
            on_token("Streamed markdown answer.")
        return {
            "final_report": "Streamed markdown answer.",
            "sources_cited": [],
            "agent_telemetry": {"total_time_ms": 50.0},
            "confidence_score": 0.9,
            "response_accuracy_score": 0.9,
            "synthesis_speedup_ratio": 0.5,
            "orchestrator": "direct",
            "model": "gemini-2.5-flash",
            "answer_origin": "cloud_synthesized",
        }

    monkeypatch.setattr(ResearchGraph, "run", fake_stream_run)

    sse = client.post(
        "/api/v1/research/stream",
        headers=dave_hdr,
        json={"query": "Stream test query", "mode": "fast", "orchestrator": "direct"},
    )
    assert sse.status_code == 200
    events = [
        json.loads(line[6:])
        for line in sse.text.splitlines()
        if line.startswith("data: ")
    ]
    complete = [ev for ev in events if ev.get("type") == "complete"]
    assert len(complete) == 1
    assert complete[0]["response"]["thread_id"]

    chats = client.get("/api/v1/history/chats", headers=dave_hdr).json()
    assert chats["total_chats"] == 1
    assert chats["chats"][0]["query"] == "Stream test query"


def test_pinned_memory_controls_and_legacy_sqlite_migration(isolated_env, monkeypatch):
    client: TestClient = isolated_env["client"]
    tmp_path: Path = isolated_env["tmp_path"]

    eve = _signup(client, "eve@example.com", "EvePass#1", "Eve").json()
    eve_hdr = {"Authorization": f"Bearer {eve['access_token']}"}

    pinned = client.post(
        "/api/v1/memory/long-term",
        headers=eve_hdr,
        json={"title": "Formatting rule", "summary": "Always include code examples.", "kind": "user_preference"},
    )
    assert pinned.status_code == 201
    mem_id = pinned.json()["memory_id"]

    mem_list = client.get("/api/v1/memory/", headers=eve_hdr).json()
    assert mem_list["long_term_count"] == 1
    assert client.delete(f"/api/v1/memory/long-term/{mem_id}", headers=eve_hdr).status_code == 200
    assert client.delete("/api/v1/memory/short-term", headers=eve_hdr).status_code == 200

    # Verify one-time automatic migration from a legacy SQLite auth.db into fresh Mongo scholar_auth DB
    legacy_db = tmp_path / "legacy_auth.db"
    conn = sqlite3.connect(str(legacy_db))
    conn.execute(
        """
        CREATE TABLE users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE, full_name TEXT,
            password_hash TEXT, role TEXT, is_active INTEGER, created_at TEXT, last_login TEXT
        )
        """
    )
    pw_hash = bcrypt.hashpw(b"Legacy#123", bcrypt.gensalt()).decode("utf-8")
    conn.execute(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("leg-1", "legacy@example.com", "Legacy User", pw_hash, "user", 1, "2026-01-01T00:00:00Z", None),
    )
    conn.commit()
    conn.close()

    mongo_mig_dir = tmp_path / "mongo_migrated"
    settings = get_settings()
    monkeypatch.setattr(settings, "AUTH_DB_PATH", str(legacy_db), raising=False)
    reset_mongo_manager(mongodb_uri="", data_dir=mongo_mig_dir, require_official=False)
    migrated_auth_db = AuthDatabase(str(legacy_db))
    migrated_user = migrated_auth_db.get_user_by_email("legacy@example.com")
    assert migrated_user is not None
    assert migrated_user["db_name"].startswith("scholar_u_")
    assert len(migrated_user["db_name"]) <= 36


def test_official_mongo_mode_refuses_silent_custom_fallback(tmp_path: Path, monkeypatch):
    from app.db.mongo_engine import MultiTenantMongoManager

    settings = get_settings()
    monkeypatch.setattr(settings, "MONGODB_URI", None, raising=False)
    monkeypatch.setattr(settings, "MONGODB_REQUIRE_OFFICIAL", True, raising=False)

    with pytest.raises(RuntimeError, match="MONGODB_REQUIRE_OFFICIAL"):
        MultiTenantMongoManager(mongodb_uri="", data_dir=tmp_path / "m1", require_official=True)

    with pytest.raises(RuntimeError, match="Refusing silent local fallback"):
        MultiTenantMongoManager(
            mongodb_uri="mongodb://127.0.0.1:59999/?serverSelectionTimeoutMS=250",
            data_dir=tmp_path / "m2",
            require_official=True,
        )
