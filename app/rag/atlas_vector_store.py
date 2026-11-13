"""Per-user chunk store backed by official MongoDB + Atlas Vector Search.

Layout
------
* Chunk text + metadata live in the user's **own** database (`<db_name>.chunks`),
  so documents stay physically separated per person.
* Embeddings live in ONE shared collection (`scholar_vectors.chunk_vectors`) covered by
  ONE Atlas `vectorSearch` index. Free (M0) clusters allow only 3 search indexes, so a
  per-user index is impossible. Every query carries a mandatory server-side pre-filter
  `owner_db == <db_name>`, and hit ids are then resolved only inside the caller's own DB,
  so a foreign chunk can never be returned.
* Lexical (BM25) retrieval keeps working from the same `chunks` collection and acts as the
  fallback while a freshly created Atlas index is still building.

The public surface mirrors `VectorStoreManager` so the retriever, routes and tools are unchanged.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from app.config import get_settings
from app.db.mongo_engine import get_mongo_manager
from app.rag.embeddings import get_embedding_model
from app.utils.exceptions import RetrievalError
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

VECTOR_COLLECTION = "chunk_vectors"
_CACHE_TTL_SECONDS = 20.0

_INDEX_LOCK = threading.Lock()
_ENSURED_INDEXES: set[tuple[int, str, str]] = set()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class AtlasVectorStore:
    def __init__(self, db_name: str, embeddings: Any = None, manager: Any = None):
        if not db_name:
            raise ValueError("User database name is required for isolated vector store access.")
        self.db_name = db_name
        self.manager = manager or get_mongo_manager()
        self.embeddings_model = embeddings
        self._embedding_attempted = embeddings is not None
        self.embedding_status = "not_loaded"
        self.warnings: List[str] = []
        self.initialized = False
        self._lock = threading.RLock()
        self._version = 0
        self._cache: Optional[List[Document]] = None
        self._cache_at = 0.0
        self._signature: Optional[Tuple[int, Any]] = None
        self.index_name = get_settings().MONGODB_VECTOR_INDEX

    # ── Collections ─────────────────────────────────────────────────────────

    @property
    def chunks(self):
        return self.manager.get_user_db(self.db_name)["chunks"]

    @property
    def vectors(self):
        return self.manager.vector_db[VECTOR_COLLECTION]

    # ── Compatibility surface ───────────────────────────────────────────────

    @property
    def dense_available(self) -> bool:
        return self.embeddings_model is not None and self.embedding_status == "ready"

    @property
    def version(self) -> int:
        self._refresh_if_stale()
        return self._version

    def initialize(self) -> None:
        with self._lock:
            if self.initialized:
                return
            self.initialized = True
            self._version += 1
            self._migrate_legacy_workspace()

    def prepare_dense(self) -> bool:
        """Load the local embedding model and make sure the Atlas vector index exists."""
        with self._lock:
            if not self._embedding_attempted:
                self._embedding_attempted = True
                try:
                    self.embeddings_model = get_embedding_model()
                except RuntimeError as exc:
                    self.embedding_status = "unavailable"
                    self.warnings.append(str(exc))
                    return False
            if self.embeddings_model is None:
                return False
            try:
                self._ensure_vector_index()
            except Exception as exc:  # index API unsupported / insufficient privileges
                self.embedding_status = "unavailable"
                warning = f"Atlas Vector Search index unavailable ({type(exc).__name__}); keyword retrieval remains available."
                if warning not in self.warnings:
                    self.warnings.append(warning)
                logger.warning(f"{warning} {exc}")
                return False
            self.embedding_status = "ready"
            return True

    # ── Atlas index management ──────────────────────────────────────────────

    def _ensure_vector_index(self) -> None:
        client = self.manager.client
        key = (id(client), self.manager.vector_db_name, self.index_name)
        with _INDEX_LOCK:
            if key in _ENSURED_INDEXES:
                return
            from pymongo.operations import SearchIndexModel

            dims = len(self.embeddings_model.embed_query("dimension probe"))
            db = self.manager.vector_db
            if VECTOR_COLLECTION not in db.list_collection_names():
                try:
                    db.create_collection(VECTOR_COLLECTION)
                except Exception:
                    pass  # created concurrently by another worker
            coll = db[VECTOR_COLLECTION]
            if not list(coll.list_search_indexes(self.index_name)):
                model = SearchIndexModel(
                    name=self.index_name,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {"type": "vector", "path": "embedding", "numDimensions": dims, "similarity": "cosine"},
                            {"type": "filter", "path": "owner_db"},
                            {"type": "filter", "path": "document_id"},
                        ]
                    },
                )
                try:
                    coll.create_search_index(model=model)
                    logger.info(f"Created Atlas vectorSearch index '{self.index_name}' ({dims} dims, cosine).")
                except Exception as exc:
                    if "already exists" not in str(exc).lower() and "duplicate" not in str(exc).lower():
                        raise
            _ENSURED_INDEXES.add(key)

    def _vector_search(self, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """Runs Atlas `$vectorSearch` pre-filtered to this user; returns (chunk_id, atlas_score)."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(k * 10, 100),
                    "limit": k,
                    "filter": {"owner_db": self.db_name},
                }
            },
            {"$project": {"_id": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]
        return [(str(row["_id"]), float(row["score"])) for row in self.vectors.aggregate(pipeline)]

    # ── Writes ──────────────────────────────────────────────────────────────

    def add_documents(self, documents) -> List[str]:
        if not documents:
            return []
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            ids: List[str] = []
            chunk_rows: List[dict] = []
            texts: List[str] = []
            for doc in documents:
                if not doc.page_content.strip():
                    continue
                chunk_id = str(uuid.uuid4())
                meta = _json_safe(dict(doc.metadata))
                chunk_rows.append({
                    "_id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id"),
                    "content": doc.page_content,
                    "metadata": meta,
                    "created_at": now,
                })
                texts.append(doc.page_content)
                ids.append(chunk_id)
            if not chunk_rows:
                return []
            self.chunks.insert_many(chunk_rows)

            if self.embeddings_model is not None:
                try:
                    embeddings = self.embeddings_model.embed_documents(texts)
                    self.vectors.insert_many([
                        {
                            "_id": row["_id"],
                            "owner_db": self.db_name,
                            "document_id": row["document_id"],
                            "embedding": [float(x) for x in vec],
                        }
                        for row, vec in zip(chunk_rows, embeddings)
                    ])
                except Exception as exc:
                    warning = "Embedding upload failed; uploaded text is preserved and keyword retrieval remains available."
                    if warning not in self.warnings:
                        self.warnings.append(warning)
                    logger.warning(f"{warning} ({exc})")
            self._invalidate()
            return ids

    def delete_documents(self, doc_ids) -> None:
        ids = [str(i) for i in doc_ids or []]
        if not ids:
            return
        with self._lock:
            self.chunks.delete_many({"_id": {"$in": ids}})
            self.vectors.delete_many({"_id": {"$in": ids}, "owner_db": self.db_name})
            self._invalidate()

    def delete_by_document_id(self, document_id: str) -> int:
        with self._lock:
            ids = [str(r["_id"]) for r in self.chunks.find({"document_id": document_id})]
            self.delete_documents(ids)
            return len(ids)

    def save(self) -> None:
        """No-op: every write is already durable in MongoDB."""
        return None

    # ── Reads ───────────────────────────────────────────────────────────────

    def get_all_documents(self) -> List[Document]:
        with self._lock:
            self._refresh_if_stale(force=self._cache is None)
            return list(self._cache or [])

    def get_document_count(self) -> int:
        return int(self.chunks.count_documents({}))

    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.25) -> List[Document]:
        if not query.strip() or k < 1 or not self.dense_available:
            return []
        try:
            query_vector = self.embeddings_model.embed_query(query)
            hits = self._vector_search(query_vector, k)
        except Exception:
            raise RetrievalError("Atlas Vector Search failed or its index is still building; keyword retrieval remains available.") from None
        if not hits:
            return []
        wanted = [cid for cid, _ in hits]
        rows = {str(r["_id"]): r for r in self.chunks.find({"_id": {"$in": wanted}})}
        results: List[Document] = []
        for chunk_id, atlas_score in hits:
            row = rows.get(chunk_id)
            if row is None:
                continue  # not in this user's own database -> never returned
            cosine = max(0.0, min(1.0, 2.0 * float(atlas_score) - 1.0))  # Atlas cosine score = (1 + cos) / 2
            if cosine >= score_threshold:
                meta = {**row.get("metadata", {}), "chunk_id": chunk_id, "dense_score": cosine, "relevance_score": cosine}
                results.append(Document(page_content=row["content"], metadata=meta))
        return results

    # ── Internals ───────────────────────────────────────────────────────────

    def _invalidate(self) -> None:
        self._cache = None
        self._cache_at = 0.0
        self._version += 1

    def _refresh_if_stale(self, force: bool = False) -> None:
        with self._lock:
            if not force and self._cache is not None and (time.monotonic() - self._cache_at) < _CACHE_TTL_SECONDS:
                return
            rows = list(self.chunks.find({}))
            signature = (len(rows), str(rows[-1]["_id"]) if rows else None)
            if self._cache is not None and signature != self._signature:
                self._version += 1  # another worker changed this user's chunks
            self._signature = signature
            self._cache = [
                Document(page_content=r["content"], metadata={**r.get("metadata", {}), "chunk_id": str(r["_id"])})
                for r in rows
            ]
            self._cache_at = time.monotonic()

    def _migrate_legacy_workspace(self) -> None:
        """One-time import of a pre-Atlas on-disk store (`documents.json`) into this user's collections."""
        try:
            legacy = Path(get_settings().USER_WORKSPACES_DIR) / self.db_name / "vector_store" / "documents.json"
            if not legacy.is_file() or self.chunks.count_documents({}) > 0:
                return
            items = json.loads(legacy.read_text(encoding="utf-8"))
            docs = [Document(page_content=i["content"], metadata=i.get("metadata", {})) for i in items]
            if docs:
                self.add_documents(docs)
            legacy.rename(legacy.with_name("documents.json.migrated"))
            logger.info(f"Migrated {len(docs)} legacy chunk(s) for {self.db_name} into MongoDB.")
        except Exception as exc:
            logger.warning(f"Legacy workspace import skipped for {self.db_name}: {exc}")
