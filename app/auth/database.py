"""MongoDB-backed persistent user & session storage with per-user database provisioning.

Stores user accounts, active/revoked JWT sessions (`jti`), and auth audit events in the
`scholar_auth` database while provisioning a dedicated private MongoDB database per user.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bcrypt

from app.db.mongo_engine import get_mongo_manager, make_user_db_name
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

MAX_FAILED_ATTEMPTS = 6
LOCKOUT_SECONDS = 60

_INIT_GUARD_LOCK = threading.Lock()
_INITIALIZED_AUTH_CLIENTS: set[tuple[int, str]] = set()


class AuthDatabase:
    """Thread-safe MongoDB manager for user records, roles, sessions, and per-user DB mapping."""

    def __init__(self, manager: Any = None):
        self._lock = threading.RLock()
        self._custom_manager = manager
        self._init_db()

    @property
    def _manager(self):
        return self._custom_manager or get_mongo_manager()

    @property
    def _users(self):
        return self._manager.auth_db["users"]

    @property
    def _sessions(self):
        return self._manager.auth_db["sessions"]

    @property
    def _audit(self):
        return self._manager.auth_db["auth_audit_logs"]

    def _init_db(self) -> None:
        client_key = (getattr(self._manager, "instance_id", id(self._manager.client)), self._manager.auth_db_name)
        if client_key in _INITIALIZED_AUTH_CLIENTS:
            return
        with _INIT_GUARD_LOCK:
            if client_key in _INITIALIZED_AUTH_CLIENTS:
                return
            self._users.create_index("email")
            self._users.create_index("id")
            self._sessions.create_index("jti")
            self._sessions.create_index("user_id")
            self._audit.create_index("user_id")

            if self._users.count_documents({}) == 0:
                self._seed_demo_admin()
            else:
                # Ensure every existing user has an isolated private MongoDB database provisioned
                for user in list(self._users.find({})):
                    if not user.get("db_name"):
                        db_name = make_user_db_name(user["id"], user.get("email", ""))
                        self._users.update_one({"id": user["id"]}, {"$set": {"db_name": db_name}})
                        self._manager.provision_user_db(db_name)
            _INITIALIZED_AUTH_CLIENTS.add(client_key)

    def _seed_demo_admin(self) -> None:
        admin_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db_name = make_user_db_name(admin_id, "admin@research.com")
        doc = {
            "_id": admin_id,
            "id": admin_id,
            "email": "admin@research.com",
            "full_name": "Lead Researcher",
            "password_hash": self.hash_password("Admin@1234"),
            "role": "admin",
            "db_name": db_name,
            "is_active": True,
            "failed_attempts": 0,
            "locked_until": None,
            "created_at": now,
            "last_login": None,
        }
        self._users.insert_one(doc)
        self._manager.provision_user_db(db_name)

    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    def ensure_user_db(self, user: Dict[str, Any]) -> str:
        db_name = user.get("db_name")
        if not db_name:
            db_name = make_user_db_name(user["id"], user.get("email", ""))
            self._users.update_one({"id": user["id"]}, {"$set": {"db_name": db_name}})
            user["db_name"] = db_name
        self._manager.provision_user_db(db_name)
        return db_name

    def create_user(self, email: str, password: str, full_name: str, role: str = "user") -> Dict[str, Any]:
        """Creates a new user and immediately provisions their private MongoDB database."""
        with self._lock:
            norm_email = email.strip().lower()
            if self._users.find_one({"email": norm_email}):
                raise ValueError(f"An account with email '{email}' already exists.")

            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            pw_hash = self.hash_password(password)
            safe_role = "admin" if str(role).lower() == "admin" else "user"
            db_name = make_user_db_name(user_id, norm_email)

            user_doc = {
                "_id": user_id,
                "id": user_id,
                "email": norm_email,
                "full_name": full_name.strip(),
                "password_hash": pw_hash,
                "role": safe_role,
                "db_name": db_name,
                "is_active": True,
                "failed_attempts": 0,
                "locked_until": None,
                "created_at": now,
                "last_login": None,
            }
            self._users.insert_one(user_doc)
            self._manager.provision_user_db(db_name)

            out = dict(user_doc)
            out.pop("_id", None)
            out.pop("password_hash", None)
            return out

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        doc = self._users.find_one({"email": email.strip().lower()})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        doc = self._users.find_one({"id": user_id})
        if doc:
            doc.pop("_id", None)
        return doc

    def is_login_locked(self, user: Dict[str, Any]) -> Optional[int]:
        locked_until = user.get("locked_until")
        if not locked_until:
            return None
        try:
            dt = datetime.fromisoformat(locked_until)
            now = datetime.now(timezone.utc)
            if dt > now:
                return max(1, int((dt - now).total_seconds()))
        except Exception:
            pass
        self._users.update_one({"id": user["id"]}, {"$set": {"failed_attempts": 0, "locked_until": None}})
        return None

    def record_failed_login(self, user_id: str) -> int:
        user = self.get_user_by_id(user_id)
        if not user:
            return 0
        attempts = int(user.get("failed_attempts", 0)) + 1
        update: Dict[str, Any] = {"failed_attempts": attempts}
        if attempts >= MAX_FAILED_ATTEMPTS:
            update["locked_until"] = (datetime.now(timezone.utc) + timedelta(seconds=LOCKOUT_SECONDS)).isoformat()
        self._users.update_one({"id": user_id}, {"$set": update})
        return attempts

    def update_last_login(self, user_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._users.update_one(
            {"id": user_id},
            {"$set": {"last_login": now, "failed_attempts": 0, "locked_until": None}},
        )

    def update_password(self, user_id: str, new_password: str, except_jti: Optional[str] = None) -> None:
        new_hash = self.hash_password(new_password)
        self._users.update_one({"id": user_id}, {"$set": {"password_hash": new_hash}})
        self.revoke_all_sessions(user_id, except_jti=except_jti)

    def update_user_admin_fields(self, user_id: str, *, role: Optional[str] = None, is_active: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        updates: Dict[str, Any] = {}
        if role in ("user", "admin"):
            updates["role"] = role
        if is_active is not None:
            updates["is_active"] = bool(is_active)
            if not is_active:
                self.revoke_all_sessions(user_id)
        if updates:
            self._users.update_one({"id": user_id}, {"$set": updates})
        u = self.get_user_by_id(user_id)
        if u:
            u.pop("password_hash", None)
        return u

    # ── Stateful Session Tracking (`jti` Revocation) ──────────────────────

    def create_session(
        self,
        *,
        jti: str,
        user_id: str,
        db_name: str,
        expires_at: str,
        user_agent: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._sessions.insert_one({
            "_id": jti,
            "jti": jti,
            "user_id": user_id,
            "db_name": db_name,
            "user_agent": (user_agent or "")[:200],
            "created_at": now,
            "expires_at": expires_at,
            "revoked": False,
            "revoked_at": None,
        })

    def is_session_active(self, jti: str, user_id: str) -> bool:
        if not jti or not user_id:
            return False
        sess = self._sessions.find_one({"jti": jti, "user_id": user_id})
        if not sess:
            return False
        return not bool(sess.get("revoked", False))

    def revoke_session(self, jti: str, user_id: Optional[str] = None) -> bool:
        filt: Dict[str, Any] = {"jti": jti}
        if user_id:
            filt["user_id"] = user_id
        now = datetime.now(timezone.utc).isoformat()
        res = self._sessions.update_one(filt, {"$set": {"revoked": True, "revoked_at": now}})
        return res.matched_count > 0

    def revoke_all_sessions(self, user_id: str, except_jti: Optional[str] = None) -> int:
        filt: Dict[str, Any] = {"user_id": user_id, "revoked": {"$ne": True}}
        if except_jti:
            filt["jti"] = {"$ne": except_jti}
        now = datetime.now(timezone.utc).isoformat()
        res = self._sessions.update_many(filt, {"$set": {"revoked": True, "revoked_at": now}})
        return int(res.modified_count)

    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        rows = list(self._sessions.find({"user_id": user_id}).sort("created_at", -1))
        for r in rows:
            r.pop("_id", None)
        return rows[:25]

    def record_activity(self, user_id: str, action: str, details: str = "") -> None:
        act_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._audit.insert_one({
            "_id": act_id,
            "id": act_id,
            "user_id": user_id,
            "action": action,
            "details": details,
            "created_at": now,
        })

    def list_users(self) -> List[Dict[str, Any]]:
        rows = list(self._users.find({}).sort("created_at", -1))
        sanitized: List[Dict[str, Any]] = []
        for r in rows:
            r.pop("_id", None)
            r.pop("password_hash", None)
            sanitized.append(r)
        return sanitized


def get_auth_db() -> AuthDatabase:
    return AuthDatabase()
