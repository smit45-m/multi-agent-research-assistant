"""End-to-end integration tests for per-user MongoDB database isolation,
short-term & long-term memory persistence, file-to-Q&A provenance, and faithful auth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.agents.graph import ResearchGraph
from app.api.routes import documents as documents_routes
from app.api.routes import research as research_routes
from app.config import get_settings
from app.db.mongo_engine import get_mongo_manager, reset_mongo_manager
from app.db.user_memory import UserMemoryService
from app.db.user_repository import UserWorkspaceRepository, clear_cached_user_vector_stores
from app.main import create_app
from app.rag.atlas_vector_store import AtlasVectorStore

_REAL_VECTOR_SEARCH = AtlasVectorStore._vector_search


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provides an in-memory MongoDB double (mongomock), isolated vector stores, and a FastAPI app."""
    workspaces_dir = tmp_path / "user_workspaces"

    settings = get_settings()
    monkeypatch.setattr(settings, "MONGODB_URI", None, raising=False)
    monkeypatch.setattr(settings, "USER_WORKSPACES_DIR", str(workspaces_dir), raising=False)

    clear_cached_user_vector_stores()
    documents_routes.DOCUMENTS.clear()
    research_routes.TASKS.clear()
    research_routes.TASK_OWNERS.clear()
    with research_routes._SYNTHESIS_CACHE_LOCK:
        research_routes._SYNTHESIS_CACHE.clear()

    mongo_client = mongomock.MongoClient()
    reset_mongo_manager(client=mongo_client)
    app = create_app()
    with TestClient(app) as client:
        yield {
            "client": client,
            "mongo_client": mongo_client,
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
    mongo_client = isolated_env["mongo_client"]

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
    assert alice_db in mongo_client.list_database_names() or mongo_client[alice_db].list_collection_names()
    assert bob_db in mongo_client.list_database_names() or mongo_client[bob_db].list_collection_names()

    mgr = get_mongo_manager()
    user_dbs = mgr.list_user_databases()
    assert alice_db in user_dbs
    assert bob_db in user_dbs
    assert mgr.auth_db_name not in user_dbs

    # Uploading a file for Alice stores chunks ONLY in Alice's own MongoDB database
    up_res = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {alice['access_token']}"},
        files={"file": ("alice_notes.txt", b"Private lattice cryptography notes for Alice.", "text/plain")},
    )
    assert up_res.status_code == 200
    assert mongo_client[alice_db]["chunks"].count_documents({}) >= 1
    assert mongo_client[bob_db]["chunks"].count_documents({}) == 0
    # Embeddings live in the shared vector collection but are tagged with their owner database
    owners = {row["owner_db"] for row in mongo_client[mgr.vector_db_name]["chunk_vectors"].find({})}
    assert owners <= {alice_db}


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
    mongo_client = isolated_env["mongo_client"]

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
    assert recorded_graph_calls[2]["user_vector_store"].get_document_count() == 0

    # Simulate complete server restart: wipe in-memory caches and reconnect to the same MongoDB server
    clear_cached_user_vector_stores()
    research_routes.TASKS.clear()
    research_routes.TASK_OWNERS.clear()
    reset_mongo_manager(client=mongo_client)
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


def test_pinned_memory_controls(isolated_env):
    client: TestClient = isolated_env["client"]

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


def test_manager_never_falls_back_to_a_hand_written_store(monkeypatch):
    from app.db import mongo_engine
    from app.db.mongo_engine import MultiTenantMongoManager

    settings = get_settings()
    monkeypatch.setattr(settings, "MONGODB_URI", None, raising=False)

    with pytest.raises(RuntimeError, match="MONGODB_URI is not configured"):
        MultiTenantMongoManager(mongodb_uri="")

    with pytest.raises(RuntimeError, match="Failed to connect to MongoDB server"):
        MultiTenantMongoManager(mongodb_uri="mongodb://127.0.0.1:59999/?serverSelectionTimeoutMS=250")

    # The removed embedded engine must stay gone
    assert not hasattr(mongo_engine, "LocalMongoClient")
    assert not hasattr(mongo_engine, "LocalMongoCollection")


def test_auth_database_has_no_sqlite_path():
    from app.auth import database as auth_database

    assert not hasattr(auth_database, "sqlite3")
    assert not hasattr(get_settings(), "AUTH_DB_PATH")
    assert not hasattr(get_settings(), "MONGODB_DATA_DIR")


def test_atlas_vector_store_isolation_delete_and_legacy_import(isolated_env):
    from langchain_core.documents import Document
    from app.rag.atlas_vector_store import AtlasVectorStore

    mgr = get_mongo_manager()
    mgr.provision_user_db("scholar_u_vec_alice")
    mgr.provision_user_db("scholar_u_vec_bob")

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text):
            low = text.lower()
            return [1.0 if "lattice" in low else 0.0, 1.0 if "mesh" in low else 0.0, 0.1]

    alice = AtlasVectorStore("scholar_u_vec_alice", embeddings=FakeEmbeddings())
    bob = AtlasVectorStore("scholar_u_vec_bob", embeddings=FakeEmbeddings())
    for store in (alice, bob):
        store.initialize()
        assert store.prepare_dense() is True and store.dense_available

    ids = alice.add_documents([
        Document(page_content="lattice cryptography notes", metadata={"document_id": "doc-a", "filename": "a.txt"}),
        Document(page_content="mesh networking notes", metadata={"document_id": "doc-b", "filename": "b.txt"}),
    ])
    bob.add_documents([Document(page_content="lattice secrets of bob", metadata={"document_id": "doc-bob"})])
    assert len(ids) == 2
    assert alice.get_document_count() == 2 and bob.get_document_count() == 1

    hits = alice.similarity_search("lattice", k=5, score_threshold=0.5)
    assert hits and hits[0].page_content == "lattice cryptography notes"
    assert all("bob" not in h.page_content for h in hits)  # strict owner isolation
    assert {d.metadata["document_id"] for d in alice.get_all_documents()} == {"doc-a", "doc-b"}

    assert alice.delete_by_document_id("doc-a") == 1
    assert alice.get_document_count() == 1
    assert not any("lattice cryptography" in h.page_content for h in alice.similarity_search("lattice", k=5, score_threshold=0.0))
    assert bob.get_document_count() == 1  # untouched

    # One-time import of a pre-Atlas on-disk workspace into MongoDB
    legacy_dir = Path(get_settings().USER_WORKSPACES_DIR) / "scholar_u_vec_carol" / "vector_store"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "documents.json").write_text(
        json.dumps([{"id": "old-1", "content": "legacy mesh chunk", "metadata": {"document_id": "doc-old"}}]),
        encoding="utf-8",
    )
    mgr.provision_user_db("scholar_u_vec_carol")
    carol = AtlasVectorStore("scholar_u_vec_carol", embeddings=FakeEmbeddings())
    carol.initialize()
    assert carol.get_document_count() == 1
    assert (legacy_dir / "documents.json.migrated").exists() and not (legacy_dir / "documents.json").exists()


def test_atlas_vector_search_pipeline_is_owner_filtered():
    from app.rag.atlas_vector_store import AtlasVectorStore

    captured = {}

    class SpyCollection:
        def aggregate(self, pipeline):
            captured["pipeline"] = pipeline
            return iter([{"_id": "c1", "score": 0.9}])

    class SpyVectorDb:
        def __getitem__(self, name):
            return SpyCollection()

    class SpyManager:
        vector_db = SpyVectorDb()
        vector_db_name = "scholar_vectors"
        client = object()

    store = AtlasVectorStore("scholar_u_spy_user", embeddings=object(), manager=SpyManager())
    hits = _REAL_VECTOR_SEARCH(store, [0.1, 0.2, 0.3], 4)
    assert hits == [("c1", 0.9)]
    stage = captured["pipeline"][0]["$vectorSearch"]
    assert stage["index"] == store.index_name
    assert stage["path"] == "embedding"
    assert stage["limit"] == 4 and stage["numCandidates"] >= 100
    assert stage["filter"] == {"owner_db": "scholar_u_spy_user"}
    assert captured["pipeline"][1]["$project"]["score"] == {"$meta": "vectorSearchScore"}


def test_writer_uses_private_user_memory_instead_of_abstaining():
    from app.agents.writer_agent import WriterAgent
    from app.agents.state import create_initial_state

    class _StubLLM:
        model = "stub-gemini"

        def __init__(self):
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append("\n".join(getattr(m, "content", str(m)) for m in messages))
            return type("Msg", (), {"content": "## Recalled Cryo Summary\n- Codename: **Project Aether-9**\n- Target: **12 mK**"})()

    stub = _StubLLM()
    writer = WriterAgent(llm=stub)

    # With private_context=True and user_memory_context populated -> MUST recall, never abstain
    state_with_mem = create_initial_state(
        "Remember what I prefer and researched earlier? Recap my cryo controller codename.",
        rag_mode="hybrid",
        orchestrator="langgraph",
        user_memory_context=(
            "Long-Term Remembered Context:\n"
            "  1. [document_insight] Cryo controller Project Aether-9 runs at 12 mK and 415 microseconds [Files: cryo.pdf]"
        ),
    )
    state_with_mem["routing_metadata"] = {"private_context": True, "per_source_chars": 2000}
    res_mem = writer.write(state_with_mem)
    assert res_mem["answer_origin"] == "llm_user_memory"
    assert "Project Aether-9" in res_mem["final_report"]
    assert "12 mK" in res_mem["final_report"]
    assert len(stub.prompts) == 1 and "personalized_user_memory" in stub.prompts[0]

    # With private_context=True and NO docs and NO user_memory_context -> MUST still safely abstain
    state_empty = create_initial_state(
        "Summarize my internal private report.",
        rag_mode="hybrid",
        orchestrator="langgraph",
        user_memory_context="",
    )
    state_empty["routing_metadata"] = {"private_context": True}
    res_empty = writer.write(state_empty)
    assert res_empty["answer_origin"] == "insufficient_evidence"
    assert res_empty["final_report"].startswith("## Not enough evidence")


def test_cross_thread_memory_recall_end_to_end_without_attachments(isolated_env, monkeypatch):
    from langchain_core.documents import Document
    from app.agents.writer_agent import WriterAgent
    from app.db.user_repository import get_user_vector_store
    from app.db.user_memory import UserMemoryService

    client: TestClient = isolated_env["client"]
    alice = _signup(client, "alice_mem@example.com", "AliceMem#1", "Alice Mem").json()
    alice_hdr = {"Authorization": f"Bearer {alice['access_token']}"}
    alice_db = alice["user"]["db_name"]
    alice_id = alice["user"]["id"]

    # Seed Alice's private vector store & turn 1 memory (preferences + PDF findings)
    uvs = get_user_vector_store(alice_db)
    uvs.add_documents([
        Document(
            page_content="Project Aether-9 superconducting cryo controller holds 12 mK target at 415 microseconds latency.",
            metadata={"filename": "cryo_spec.pdf", "title": "cryo_spec.pdf", "document_id": "doc-cryo-1", "format": "pdf"},
        )
    ])
    UserMemoryService(alice_db, alice_id).update_after_turn(
        chat_id="turn-1",
        thread_id="thread-alpha",
        query="Please always respond in concise bullet points. Summarize Project Aether-9.",
        report="Project Aether-9 maintains a 12 mK target temperature and 415 microseconds cryo control loop.",
        attachment_names=["cryo_spec.pdf"],
        sources=[{"title": "cryo_spec.pdf"}],
    )

    seen_payloads = []

    def fake_call_model(llm, state, system, prompt, **kwargs):
        seen_payloads.append(prompt)
        return "- **Cryo Controller**: Project Aether-9\n- **Temperature Target**: 12 mK\n- **Preference**: Concise bullets"

    import app.agents.writer_agent as wa_mod
    monkeypatch.setattr(wa_mod, "call_model", fake_call_model)

    res = client.post(
        "/api/v1/research/sync",
        headers=alice_hdr,
        json={
            "query": "Remember what I prefer and researched earlier? Briefly recap my cryo controller codename and temperature target.",
            "mode": "research",
            "web_search": False,
            "thread_id": "brand-new-thread-beta",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert not body["report"].startswith("## Not enough evidence")
    assert "Project Aether-9" in body["report"]
    assert "12 mK" in body["report"]
    assert body["memory_context_used"]["short_term_count"] >= 1 or body["memory_context_used"]["long_term_count"] >= 1
    assert seen_payloads and "Project Aether-9" in seen_payloads[0]


def test_research_mode_with_standard_word_not_hijacked_by_nda_substring():
    from app.agents.meta_agent import MetaAgent

    cfg = MetaAgent().analyze_and_configure(
        "Compare FlashAttention vs standard Transformer self-attention in HBM memory complexity, SRAM tiling, and exactness.",
        mode="research",
        depth="deep",
        privacy_mode=False,
    )
    assert cfg["resolved_mode"] == "research"
    assert cfg["selected_orchestrator"] == "langgraph"

    # Whole-word 'NDA' in auto mode still triggers privacy protection
    cfg_nda = MetaAgent().analyze_and_configure(
        "Review our confidential NDA merger memo.",
        mode="auto",
        privacy_mode=False,
    )
    assert cfg_nda["resolved_mode"] == "privacy"


def test_supervisor_does_not_inject_hybrid_rag_for_researcher_or_memory_queries():
    from app.agents.state import create_initial_state
    from app.agents.supervisor_agent import SupervisorAgent

    sup = SupervisorAgent(llm=None)
    state = create_initial_state(
        "I am a quantum cryogenics researcher. Briefly recap what I researched earlier about my cryo target.",
        rag_mode="hybrid",
        orchestrator="langgraph",
        user_memory_context="Remembered Facts:\n- CryoPulse target: 850 mW @ 4K",
    )
    state["mode"] = "fast"
    state["answer_origin"] = "llm_user_memory"
    state["final_report"] = "- **CryoPulse Target**: 850 mW @ 4K (`cryo.pdf`)."
    state["offline"] = True

    out = sup.supervise(state)
    assert "Hybrid RAG" not in out["final_report"]
    assert "CryoPulse Target" in out["final_report"]

    # Direct comparison table builder also respects word boundaries ('researcher' != 'search')
    table = sup._build_comparison_table("I am a researcher discussing cryo cooling", [])
    assert "Hybrid RAG" not in table


def test_bounded_llm_strips_zero_width_and_skips_truncated_thinking_stub(monkeypatch):
    import time
    from app.chains.llm import BoundedLLM, _clean_model_text
    from app.config import get_settings

    assert _clean_model_text("\u200b \ufeff\n ") == ""

    calls = []

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(url)
            if "gemini-flash-lite-latest" in url:
                # Simulate an invisible \u200b placeholder that must be skipped
                return FakeResp(200, {
                    "candidates": [{
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "\u200b"}]},
                    }]
                })
            if "gemini-3.5-flash-lite" in url:
                return FakeResp(200, {
                    "candidates": [{
                        "finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": "Your SPIFFE/SPIRE X.5"}]},
                    }]
                })
            return FakeResp(200, {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [
                        {"text": "internal thought", "thought": True},
                        {"text": "- **SPIFFE/SPIRE SVID**: Zero-trust mTLS identity with 12-hour X.509 rotation and OPA enforcement."},
                    ]},
                }],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 25},
            })

    import app.chains.llm as llm_mod
    monkeypatch.setattr(llm_mod, "get_shared_client", lambda: FakeClient())
    settings = get_settings().model_copy(update={
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "test-gemini-key",
    })
    monkeypatch.setattr(llm_mod, "get_settings", lambda: settings)

    llm = BoundedLLM(mode="fast", deadline=time.monotonic() + 10.0)
    out = llm.complete("system", "prompt", max_tokens=1000)
    assert "internal thought" not in out
    assert "SPIFFE/SPIRE SVID" in out
    assert len(calls) == 3
