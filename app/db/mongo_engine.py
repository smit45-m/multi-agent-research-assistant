"""MongoDB multi-database engine (live PyMongo + durable embedded per-DB document engine).

Each registered user gets their own physically separated MongoDB database
(e.g. `scholar_u_alice_1a2b3c4d`), isolating chats, Q&As, uploaded files,
short-term context, and long-term memories across users.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_DB_NAME_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def make_user_db_name(user_id: str, email: str = "") -> str:
    """Builds a deterministic, filesystem- and MongoDB-safe per-user DB name (<= 36 chars)."""
    local_part = (email.split("@")[0] if "@" in email else email or "user").lower()
    slug = _DB_NAME_SAFE_RE.sub("_", local_part).strip("_")[:12] or "user"
    uid_clean = _DB_NAME_SAFE_RE.sub("", str(user_id).lower())[:12] or uuid.uuid4().hex[:12]
    return f"scholar_u_{slug}_{uid_clean}"[:36]


def _get_nested(doc: Dict[str, Any], dotted_key: str) -> Tuple[bool, Any]:
    parts = dotted_key.split(".")
    cur: Any = doc
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return False, None
    return True, cur


def _set_nested(doc: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = doc
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = copy.deepcopy(value)


def _unset_nested(doc: Dict[str, Any], dotted_key: str) -> None:
    parts = dotted_key.split(".")
    cur: Any = doc
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _match_condition(val: Any, exists: bool, cond: Any) -> bool:
    if isinstance(cond, dict) and any(str(k).startswith("$") for k in cond.keys()):
        for op, target in cond.items():
            if op == "$eq":
                if val != target:
                    return False
            elif op == "$ne":
                if val == target:
                    return False
            elif op == "$in":
                if isinstance(val, list):
                    if not any(v in target for v in val):
                        return False
                elif val not in target:
                    return False
            elif op == "$nin":
                if isinstance(val, list):
                    if any(v in target for v in val):
                        return False
                elif val in target:
                    return False
            elif op == "$gt":
                if not exists or val is None or not (val > target):
                    return False
            elif op == "$gte":
                if not exists or val is None or not (val >= target):
                    return False
            elif op == "$lt":
                if not exists or val is None or not (val < target):
                    return False
            elif op == "$lte":
                if not exists or val is None or not (val <= target):
                    return False
            elif op == "$exists":
                if bool(exists) != bool(target):
                    return False
            elif op == "$regex":
                flags = re.IGNORECASE if "i" in str(cond.get("$options", "")) else 0
                if not exists or val is None:
                    return False
                if not re.search(str(target), str(val), flags):
                    return False
            elif op == "$options":
                continue
        return True

    if isinstance(val, list) and not isinstance(cond, list):
        return cond in val
    return exists and val == cond


def _matches_filter(doc: Dict[str, Any], filt: Optional[Dict[str, Any]]) -> bool:
    if not filt:
        return True
    for k, cond in filt.items():
        if k == "$and":
            if not all(_matches_filter(doc, sub) for sub in cond):
                return False
        elif k == "$or":
            if not any(_matches_filter(doc, sub) for sub in cond):
                return False
        elif k == "$nor":
            if any(_matches_filter(doc, sub) for sub in cond):
                return False
        else:
            exists, val = _get_nested(doc, k)
            if not _match_condition(val, exists, cond):
                return False
    return True


def _apply_update(doc: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    has_ops = any(str(k).startswith("$") for k in update.keys())
    if not has_ops:
        new_doc = copy.deepcopy(update)
        if "_id" not in new_doc and "_id" in doc:
            new_doc["_id"] = doc["_id"]
        return new_doc

    for op, fields in update.items():
        if op == "$set" and isinstance(fields, dict):
            for k, v in fields.items():
                _set_nested(doc, k, v)
        elif op == "$unset" and isinstance(fields, dict):
            for k in fields.keys():
                _unset_nested(doc, k)
        elif op == "$inc" and isinstance(fields, dict):
            for k, delta in fields.items():
                exists, cur = _get_nested(doc, k)
                base = cur if (exists and isinstance(cur, (int, float))) else 0
                _set_nested(doc, k, base + delta)
        elif op == "$push" and isinstance(fields, dict):
            for k, item in fields.items():
                exists, cur = _get_nested(doc, k)
                arr = list(cur) if (exists and isinstance(cur, list)) else []
                if isinstance(item, dict) and "$each" in item:
                    arr.extend(copy.deepcopy(item["$each"]))
                    sl = item.get("$slice")
                    if isinstance(sl, int):
                        arr = arr[sl:] if sl < 0 else arr[:sl]
                else:
                    arr.append(copy.deepcopy(item))
                _set_nested(doc, k, arr)
        elif op == "$addToSet" and isinstance(fields, dict):
            for k, item in fields.items():
                exists, cur = _get_nested(doc, k)
                arr = list(cur) if (exists and isinstance(cur, list)) else []
                to_add = item["$each"] if (isinstance(item, dict) and "$each" in item) else [item]
                for x in to_add:
                    if x not in arr:
                        arr.append(copy.deepcopy(x))
                _set_nested(doc, k, arr)
        elif op == "$pull" and isinstance(fields, dict):
            for k, item in fields.items():
                exists, cur = _get_nested(doc, k)
                if exists and isinstance(cur, list):
                    if isinstance(item, dict):
                        _set_nested(doc, k, [el for el in cur if not (isinstance(el, dict) and _matches_filter(el, item))])
                    else:
                        _set_nested(doc, k, [el for el in cur if el != item])
    return doc


@dataclass
class _InsertOneResult:
    inserted_id: Any
    acknowledged: bool = True


@dataclass
class _InsertManyResult:
    inserted_ids: List[Any]
    acknowledged: bool = True


@dataclass
class _UpdateResult:
    matched_count: int
    modified_count: int
    upserted_id: Any = None
    acknowledged: bool = True


@dataclass
class _DeleteResult:
    deleted_count: int
    acknowledged: bool = True


class LocalMongoCursor:
    """Chainable PyMongo-compatible cursor supporting sort, skip, limit, and iteration."""

    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs
        self._sort_specs: List[Tuple[str, int]] = []
        self._skip = 0
        self._limit: Optional[int] = None

    def sort(self, key_or_list: Union[str, List[Tuple[str, int]]], direction: Optional[int] = None) -> "LocalMongoCursor":
        if isinstance(key_or_list, str):
            self._sort_specs = [(key_or_list, direction if direction is not None else 1)]
        else:
            self._sort_specs = list(key_or_list)
        return self

    def skip(self, count: int) -> "LocalMongoCursor":
        self._skip = max(0, int(count))
        return self

    def limit(self, count: int) -> "LocalMongoCursor":
        self._limit = int(count) if count and count > 0 else None
        return self

    def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = list(self._docs)
        for key, dir_val in reversed(self._sort_specs):
            rows.sort(
                key=lambda d: (0, "") if not _get_nested(d, key)[0] or _get_nested(d, key)[1] is None else (1, str(_get_nested(d, key)[1])),
                reverse=(dir_val < 0),
            )
        if self._skip:
            rows = rows[self._skip:]
        cap = self._limit if length is None else (min(self._limit, length) if self._limit else length)
        if cap is not None:
            rows = rows[:cap]
        return [copy.deepcopy(r) for r in rows]

    def __iter__(self):
        return iter(self.to_list())


class LocalMongoCollection:
    """Thread-safe disk-persisted JSON document collection per database."""

    def __init__(self, db_dir: Path, name: str):
        self.name = name
        self._file_path = db_dir / f"{name}.json"
        self._lock = threading.RLock()
        self._docs: List[Dict[str, Any]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._file_path.exists():
            try:
                raw = self._file_path.read_text(encoding="utf-8")
                data = json.loads(raw) if raw.strip() else []
                if isinstance(data, list):
                    self._docs = data
            except Exception as exc:
                logger.warning(f"Failed loading collection {self._file_path}: {exc}")
                self._docs = []
        self._loaded = True

    def _flush(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file_path.parent / f".{self.name}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(self._docs, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        os.replace(tmp, self._file_path)

    def create_index(self, *args, **kwargs) -> str:
        # Ensure collection storage exists cleanly
        with self._lock:
            self._ensure_loaded()
            if not self._file_path.exists():
                self._flush()
        return f"idx_{self.name}"

    def count_documents(self, filter: Optional[Dict[str, Any]] = None) -> int:
        with self._lock:
            self._ensure_loaded()
            return sum(1 for d in self._docs if _matches_filter(d, filter))

    def find_one(self, filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._ensure_loaded()
            for d in self._docs:
                if _matches_filter(d, filter):
                    return copy.deepcopy(d)
            return None

    def find(self, filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None) -> LocalMongoCursor:
        with self._lock:
            self._ensure_loaded()
            matched = [copy.deepcopy(d) for d in self._docs if _matches_filter(d, filter)]
            return LocalMongoCursor(matched)

    def insert_one(self, document: Dict[str, Any]) -> _InsertOneResult:
        with self._lock:
            self._ensure_loaded()
            doc = copy.deepcopy(document)
            doc.setdefault("_id", str(uuid.uuid4()))
            self._docs.append(doc)
            self._flush()
            return _InsertOneResult(inserted_id=doc["_id"])

    def insert_many(self, documents: Iterable[Dict[str, Any]]) -> _InsertManyResult:
        with self._lock:
            self._ensure_loaded()
            ids = []
            for item in documents:
                doc = copy.deepcopy(item)
                doc.setdefault("_id", str(uuid.uuid4()))
                self._docs.append(doc)
                ids.append(doc["_id"])
            self._flush()
            return _InsertManyResult(inserted_ids=ids)

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> _UpdateResult:
        with self._lock:
            self._ensure_loaded()
            for idx, d in enumerate(self._docs):
                if _matches_filter(d, filter):
                    self._docs[idx] = _apply_update(copy.deepcopy(d), update)
                    self._flush()
                    return _UpdateResult(matched_count=1, modified_count=1)
            if upsert:
                base: Dict[str, Any] = {}
                for k, v in (filter or {}).items():
                    if not str(k).startswith("$") and not isinstance(v, dict):
                        _set_nested(base, k, v)
                base.setdefault("_id", str(uuid.uuid4()))
                base = _apply_update(base, update)
                self._docs.append(base)
                self._flush()
                return _UpdateResult(matched_count=0, modified_count=0, upserted_id=base["_id"])
            return _UpdateResult(matched_count=0, modified_count=0)

    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any]) -> _UpdateResult:
        with self._lock:
            self._ensure_loaded()
            matched = 0
            for idx, d in enumerate(self._docs):
                if _matches_filter(d, filter):
                    self._docs[idx] = _apply_update(copy.deepcopy(d), update)
                    matched += 1
            if matched:
                self._flush()
            return _UpdateResult(matched_count=matched, modified_count=matched)

    def delete_one(self, filter: Dict[str, Any]) -> _DeleteResult:
        with self._lock:
            self._ensure_loaded()
            for idx, d in enumerate(self._docs):
                if _matches_filter(d, filter):
                    del self._docs[idx]
                    self._flush()
                    return _DeleteResult(deleted_count=1)
            return _DeleteResult(deleted_count=0)

    def delete_many(self, filter: Dict[str, Any]) -> _DeleteResult:
        with self._lock:
            self._ensure_loaded()
            before = len(self._docs)
            self._docs = [d for d in self._docs if not _matches_filter(d, filter)]
            deleted = before - len(self._docs)
            if deleted:
                self._flush()
            return _DeleteResult(deleted_count=deleted)


class LocalMongoDatabase:
    """Isolated per-database directory holding collections for one tenant/user or auth plane."""

    def __init__(self, base_dir: Path, name: str):
        self.name = name
        self._db_dir = base_dir / name
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._collections: Dict[str, LocalMongoCollection] = {}

    def __getitem__(self, collection_name: str) -> LocalMongoCollection:
        return self.get_collection(collection_name)

    def get_collection(self, collection_name: str) -> LocalMongoCollection:
        with self._lock:
            if collection_name not in self._collections:
                self._collections[collection_name] = LocalMongoCollection(self._db_dir, collection_name)
            return self._collections[collection_name]

    def list_collection_names(self) -> List[str]:
        if not self._db_dir.exists():
            return []
        return sorted(p.stem for p in self._db_dir.glob("*.json"))


class LocalMongoClient:
    """MongoClient-compatible multi-database store persisted under separate per-DB folders."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._dbs: Dict[str, LocalMongoDatabase] = {}

    def __getitem__(self, db_name: str) -> LocalMongoDatabase:
        return self.get_database(db_name)

    def get_database(self, db_name: str) -> LocalMongoDatabase:
        with self._lock:
            if db_name not in self._dbs:
                self._dbs[db_name] = LocalMongoDatabase(self.base_dir, db_name)
            return self._dbs[db_name]

    def list_database_names(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(p.name for p in self.base_dir.iterdir() if p.is_dir())

    def drop_database(self, db_name: str) -> None:
        with self._lock:
            self._dbs.pop(db_name, None)
            target = self.base_dir / db_name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)


_MANAGER_SEQ = 0


class MultiTenantMongoManager:
    """
    Central coordinator creating one isolated MongoDB database per registered user.
    Connects to official MongoDB Atlas/Server via PyMongo Stable API v1 when `MONGODB_URI` is set,
    and refuses silent fallback to the local test engine when `require_official` is active.
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        data_dir: Optional[Union[str, Path]] = None,
        require_official: Optional[bool] = None,
    ):
        global _MANAGER_SEQ
        settings = get_settings()
        _MANAGER_SEQ += 1
        self.instance_id = _MANAGER_SEQ
        self.auth_db_name = getattr(settings, "MONGODB_AUTH_DB", "scholar_auth")
        self.uri = mongodb_uri if mongodb_uri is not None else getattr(settings, "MONGODB_URI", None)
        cfg_strict = bool(getattr(settings, "MONGODB_REQUIRE_OFFICIAL", False))
        # Whenever an explicit URI is configured, fail loudly instead of silently switching backends
        self.require_official = require_official if require_official is not None else (cfg_strict or bool(self.uri))
        self.data_dir = Path(data_dir or getattr(settings, "MONGODB_DATA_DIR", "data/mongo_dbs"))
        self.backend = "embedded_mongo"
        self.client: Any = None
        self._lock = threading.RLock()
        self._provisioned_dbs: set[str] = set()
        self._init_client()

    def _init_client(self) -> None:
        if self.uri:
            try:
                import certifi
                from pymongo import MongoClient
                from pymongo.server_api import ServerApi

                kwargs: Dict[str, Any] = {
                    "server_api": ServerApi("1"),
                }
                if "serverselectiontimeoutms" not in self.uri.lower():
                    kwargs["serverSelectionTimeoutMS"] = 8000
                if "mongodb+srv://" in self.uri or "tls=true" in self.uri.lower() or "ssl=true" in self.uri.lower():
                    kwargs["tlsCAFile"] = certifi.where()

                cand = MongoClient(self.uri, **kwargs)
                cand.admin.command("ping")
                self.client = cand
                self.backend = "pymongo"
                logger.info("Connected to official MongoDB Server via PyMongo (Stable API v1).")
                return
            except Exception as exc:
                if self.require_official:
                    raise RuntimeError(
                        f"Failed to connect to official MongoDB Server configured via MONGODB_URI: {exc}. "
                        "Refusing silent local fallback because official MongoDB mode is active."
                    ) from exc
                logger.warning(f"MongoDB server unreachable at configured URI ({exc}); using test fallback.")

        if self.require_official:
            raise RuntimeError(
                "MONGODB_REQUIRE_OFFICIAL=true but MONGODB_URI is not configured. "
                "Set MONGODB_URI in .env to your official MongoDB Atlas or mongod URI."
            )

        self.client = LocalMongoClient(self.data_dir)
        self.backend = "embedded_mongo"
        logger.info(f"Initialized isolated test/local Mongo engine at {self.data_dir}")

    @property
    def auth_db(self):
        return self.client[self.auth_db_name]

    def provision_user_db(self, db_name: str):
        if not db_name or db_name == self.auth_db_name:
            raise ValueError("Refusing to bind user workspace to control-plane auth database.")
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
            if d not in (self.auth_db_name, "admin", "local", "config")
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


def reset_mongo_manager(
    mongodb_uri: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
    require_official: Optional[bool] = None,
) -> MultiTenantMongoManager:
    global _MANAGER_INSTANCE
    with _MANAGER_LOCK:
        _MANAGER_INSTANCE = MultiTenantMongoManager(
            mongodb_uri=mongodb_uri,
            data_dir=data_dir,
            require_official=require_official,
        )
        return _MANAGER_INSTANCE
