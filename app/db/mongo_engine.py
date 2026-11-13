"""Official MongoDB multi-database engine (PyMongo, Stable API v1, Atlas).

Each registered user gets their own physically separated MongoDB database
(e.g. `scholar_u_alice_1a2b3c4d`), isolating chats, Q&As, uploaded file metadata,
short-term context, and long-term memories across users.

There is intentionally no embedded/local fallback: when `MONGODB_URI` is missing
or the server is unreachable the application fails loudly instead of silently
switching to a hand-written store.
"""
from __future__ import annotations

import re
import threading
import uuid
from typing import Any, List, Optional

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_DB_NAME_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def make_user_db_name(user_id: str, email: str = "") -> str:
    """Builds a deterministic MongoDB-safe per-user DB name (<= 36 chars)."""
    local_part = (email.split("@")[0] if "@" in email else email or "user").lower()
    slug = _DB_NAME_SAFE_RE.sub("_", local_part).strip("_")[:12] or "user"
    uid_clean = _DB_NAME_SAFE_RE.sub("", str(user_id).lower())[:12] or uuid.uuid4().hex[:12]
    return f"scholar_u_{slug}_{uid_clean}"[:36]


_MANAGER_SEQ = 0


class MultiTenantMongoManager:
    """
    Central coordinator creating one isolated MongoDB database per registered user.

    Connects to the official MongoDB server (Atlas) via PyMongo Stable API v1 using
    `MONGODB_URI`. There is no silent fallback: a missing URI or unreachable server
    raises `RuntimeError`. A pre-built client may be injected (used by unit tests).
    """

    def __init__(self, mongodb_uri: Optional[str] = None, client: Any = None):
        global _MANAGER_SEQ
        settings = get_settings()
        _MANAGER_SEQ += 1
        self.instance_id = _MANAGER_SEQ
        self.auth_db_name = getattr(settings, "MONGODB_AUTH_DB", "scholar_auth")
        self.uri = mongodb_uri if mongodb_uri is not None else getattr(settings, "MONGODB_URI", None)
        self.backend = "pymongo"
        self.client: Any = None
        self._lock = threading.RLock()
        self._provisioned_dbs: set[str] = set()
        if client is not None:
            self.client = client
            self.backend = "injected"
        else:
            self._connect()

    def _connect(self) -> None:
        if not self.uri:
            raise RuntimeError(
                "MONGODB_URI is not configured. Set MONGODB_URI in .env to your official "
                "MongoDB Atlas (or mongod) connection string."
            )
        try:
            import certifi
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi

            kwargs: dict[str, Any] = {"server_api": ServerApi("1")}
            if "serverselectiontimeoutms" not in self.uri.lower():
                kwargs["serverSelectionTimeoutMS"] = 8000
            if "mongodb+srv://" in self.uri or "tls=true" in self.uri.lower() or "ssl=true" in self.uri.lower():
                kwargs["tlsCAFile"] = certifi.where()

            client = MongoClient(self.uri, **kwargs)
            client.admin.command("ping")
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to MongoDB server configured via MONGODB_URI: {exc}") from exc
        self.client = client
        logger.info("Connected to official MongoDB Server via PyMongo (Stable API v1).")

    @property
    def auth_db(self):
        return self.client[self.auth_db_name]

    def provision_user_db(self, db_name: str):
        if not db_name or db_name == self.auth_db_name:
            raise ValueError("Refusing to bind user workspace to the shared auth database.")
        with self._lock:
            db = self.client[db_name]
            if db_name not in self._provisioned_dbs:
                db["chats"].create_index("created_at")
                db["chats"].create_index("thread_id")
                db["files"].create_index("document_id")
                db["short_term_memory"].create_index("created_at")
                db["long_term_memory"].create_index("topic_key")
                self._provisioned_dbs.add(db_name)
            return db

    def get_user_db(self, db_name: str):
        if not db_name or db_name == self.auth_db_name:
            raise ValueError("Invalid or unauthorized per-user database target.")
        return self.client[db_name]

    def list_user_databases(self) -> List[str]:
        return [
            d
            for d in self.client.list_database_names()
            if d.startswith("scholar_u_")
        ]


_MANAGER_LOCK = threading.Lock()
_MANAGER_INSTANCE: Optional[MultiTenantMongoManager] = None


def get_mongo_manager() -> MultiTenantMongoManager:
    global _MANAGER_INSTANCE
    if _MANAGER_INSTANCE is None:
        with _MANAGER_LOCK:
            if _MANAGER_INSTANCE is None:
                _MANAGER_INSTANCE = MultiTenantMongoManager()
    return _MANAGER_INSTANCE


def reset_mongo_manager(mongodb_uri: Optional[str] = None, client: Any = None) -> MultiTenantMongoManager:
    global _MANAGER_INSTANCE
    with _MANAGER_LOCK:
        _MANAGER_INSTANCE = MultiTenantMongoManager(mongodb_uri=mongodb_uri, client=client)
        return _MANAGER_INSTANCE
