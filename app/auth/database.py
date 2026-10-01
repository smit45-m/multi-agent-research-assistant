"""SQLite-backed persistent user storage with bcrypt hashing and transaction safety."""
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import bcrypt

from app.config import get_settings


class AuthDatabase:
    """Thread-safe SQLite manager for user records, roles, and auth activity."""

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(AuthDatabase, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        settings = get_settings()
        self.db_path = Path(db_path or settings.AUTH_DB_PATH)
        self._lock = threading.Lock()
        self._init_db()
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        last_login TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_activity (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                conn.commit()

                # Seed initial demo admin account if database is freshly created
                cursor.execute("SELECT COUNT(*) as cnt FROM users")
                row = cursor.fetchone()
                if row and row["cnt"] == 0:
                    admin_id = str(uuid.uuid4())
                    now = datetime.now(timezone.utc).isoformat()
                    demo_hash = self.hash_password("Admin@1234")
                    cursor.execute("""
                        INSERT INTO users (id, email, full_name, password_hash, role, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?, 1, ?)
                    """, (admin_id, "admin@research.com", "Lead Researcher", demo_hash, "admin", now))
                    conn.commit()
            finally:
                conn.close()

    @staticmethod
    def hash_password(password: str) -> str:
        """Hashes password using bcrypt with standard salt rounds."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifies plaintext password against stored bcrypt hash."""
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    def create_user(self, email: str, password: str, full_name: str, role: str = "user") -> Dict[str, Any]:
        """Creates a new user record. Raises ValueError if email is taken."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                norm_email = email.strip().lower()
                cursor.execute("SELECT id FROM users WHERE email = ?", (norm_email,))
                if cursor.fetchone():
                    raise ValueError(f"An account with email '{email}' already exists.")

                user_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                pw_hash = self.hash_password(password)
                safe_role = "admin" if role.lower() == "admin" else "user"

                cursor.execute("""
                    INSERT INTO users (id, email, full_name, password_hash, role, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (user_id, norm_email, full_name.strip(), pw_hash, safe_role, now))
                conn.commit()

                return {
                    "id": user_id,
                    "email": norm_email,
                    "full_name": full_name.strip(),
                    "role": safe_role,
                    "is_active": True,
                    "created_at": now,
                    "last_login": None
                }
            finally:
                conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def update_last_login(self, user_id: str) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))
                conn.commit()
            finally:
                conn.close()

    def record_activity(self, user_id: str, action: str, details: str = "") -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                act_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    INSERT INTO user_activity (id, user_id, action, details, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (act_id, user_id, action, details, now))
                conn.commit()
            finally:
                conn.close()

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, email, full_name, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC")
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()


def get_auth_db() -> AuthDatabase:
    return AuthDatabase()
