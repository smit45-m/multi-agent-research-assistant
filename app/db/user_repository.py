"""Per-user isolated MongoDB repository and dedicated vector store manager."""
from __future__ import annotations

import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.mongo_engine import get_mongo_manager
from app.rag.atlas_vector_store import AtlasVectorStore
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_VECTOR_STORE_LOCK = threading.RLock()
_USER_VECTOR_STORES: Dict[str, AtlasVectorStore] = {}
_SHARED_EMBEDDINGS_LOCK = threading.Lock()
_SHARED_EMBEDDINGS: Any = None
_SHARED_EMBEDDINGS_TRIED = False


def _get_shared_embeddings():
    global _SHARED_EMBEDDINGS, _SHARED_EMBEDDINGS_TRIED
    if _SHARED_EMBEDDINGS_TRIED:
        return _SHARED_EMBEDDINGS
    with _SHARED_EMBEDDINGS_LOCK:
        if not _SHARED_EMBEDDINGS_TRIED:
            _SHARED_EMBEDDINGS_TRIED = True
            try:
                from app.rag.embeddings import get_embedding_model
                _SHARED_EMBEDDINGS = get_embedding_model()
            except Exception as exc:
                logger.warning(f"Shared local embedding model unavailable ({exc}); using BM25 lexical index.")
                _SHARED_EMBEDDINGS = None
    return _SHARED_EMBEDDINGS


def get_user_vector_store(db_name: str) -> AtlasVectorStore:
    """Returns the Atlas-backed chunk/vector store scoped strictly to `db_name`."""
    if not db_name:
        raise ValueError("User database name is required for isolated vector store access.")
    with _VECTOR_STORE_LOCK:
        existing = _USER_VECTOR_STORES.get(db_name)
        if existing is not None:
            return existing
        emb = _get_shared_embeddings()
        store = AtlasVectorStore(db_name, embeddings=emb)
        store.initialize()
        if emb is not None:
            store.prepare_dense()
        _USER_VECTOR_STORES[db_name] = store
        return store


def clear_cached_user_vector_stores() -> None:
    """Clears in-memory user vector store cache (useful in isolated tests)."""
    with _VECTOR_STORE_LOCK:
        _USER_VECTOR_STORES.clear()


def _summarize_answer(report: str, max_len: int = 380) -> str:
    if not report:
        return ""
    lines = []
    for raw in report.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">") and ("Air-Gapped" in s or "Privacy" in s or "Synthesized via" in s):
            continue
        if s.startswith("# 🔒"):
            continue
        cleaned = re.sub(r"^#+\s*", "", s)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        if cleaned:
            lines.append(cleaned)
        if sum(len(x) for x in lines) >= max_len:
            break
    joined = " ".join(lines).strip()
    return joined[:max_len] + ("..." if len(joined) > max_len else "")


class UserWorkspaceRepository:
    """All reads and writes stay strictly confined to one authenticated user's MongoDB DB."""

    def __init__(self, db_name: str, user_id: str = ""):
        self.db_name = db_name
        self.user_id = user_id
        self.manager = get_mongo_manager()
        self.db = self.manager.get_user_db(db_name)
        self.chats = self.db["chats"]
        self.files = self.db["files"]

    # ── File Tracking & File-to-Q&A Linking ────────────────────────────────

    def record_file_upload(
        self,
        *,
        document_id: str,
        filename: str,
        fmt: str,
        chunk_count: int,
        size_bytes: int = 0,
        sha256: str = "",
        preview_text: str = "",
        privacy_mode: bool = False,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "document_id": document_id,
            "user_id": self.user_id,
            "filename": filename,
            "format": fmt.lstrip(".").lower() or "document",
            "chunk_count": int(chunk_count),
            "size_bytes": int(size_bytes),
            "sha256": sha256,
            "preview_text": (preview_text or "")[:500],
            "privacy_processed": bool(privacy_mode),
            "status": "processed",
            "uploaded_at": now,
            "deleted": False,
            "qa_count": 0,
            "qa_links": [],
        }
        self.files.update_one({"document_id": document_id}, {"$set": record}, upsert=True)
        return dict(record)

    def list_files(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        filt = {} if include_deleted else {"deleted": {"$ne": True}}
        docs = list(self.files.find(filt).sort("uploaded_at", -1))
        for d in docs:
            d.pop("_id", None)
            links = d.get("qa_links") or []
            d["qa_count"] = len(links)
        return docs

    def get_file(self, document_id: str) -> Optional[Dict[str, Any]]:
        doc = self.files.find_one({"document_id": document_id, "deleted": {"$ne": True}})
        if doc:
            doc.pop("_id", None)
        return doc

    def delete_file(self, document_id: str) -> bool:
        existing = self.files.find_one({"document_id": document_id, "deleted": {"$ne": True}})
        if not existing:
            # Allow matching by filename when user passes filename
            existing = self.files.find_one({"filename": document_id, "deleted": {"$ne": True}})
        if not existing:
            return False
        target_id = existing["document_id"]
        self.files.update_one(
            {"document_id": target_id},
            {"$set": {"deleted": True, "deleted_at": datetime.now(timezone.utc).isoformat()}},
        )
        try:
            vs = get_user_vector_store(self.db_name)
            removed = vs.delete_by_document_id(target_id)
            if removed == 0:
                fname = existing.get("filename")
                if fname:
                    ids_to_del = [
                        d.metadata["chunk_id"]
                        for d in vs.get_all_documents()
                        if d.metadata.get("filename") == fname and d.metadata.get("chunk_id")
                    ]
                    if ids_to_del:
                        vs.delete_documents(ids_to_del)
        except Exception as exc:
            logger.warning(f"Could not purge vector store chunks on user file delete ({target_id}): {exc}")
        return True

    def _link_files_to_qa(
        self,
        *,
        chat_id: str,
        thread_id: str,
        query: str,
        answer_summary: str,
        created_at: str,
        attachment_ids: List[str],
        sources: List[Dict[str, Any]],
    ) -> List[str]:
        """Associates every user-owned file used/cited in this Q&A turn."""
        all_user_files = list(self.files.find({"deleted": {"$ne": True}}))
        if not all_user_files:
            return []

        by_doc_id = {f["document_id"]: f for f in all_user_files}
        by_filename = {str(f.get("filename", "")).lower(): f for f in all_user_files if f.get("filename")}
        matched_ids = set()

        for aid in attachment_ids or []:
            if aid in by_doc_id:
                matched_ids.add(aid)

        for s in sources or []:
            src_url = str(s.get("url_or_path") or s.get("source") or "")
            src_title = str(s.get("title") or "").lower()
            if src_url.startswith("attachment:"):
                aid = src_url.split("attachment:", 1)[1]
                if aid in by_doc_id:
                    matched_ids.add(aid)
            if src_title in by_filename:
                matched_ids.add(by_filename[src_title]["document_id"])
            for fname_low, fdoc in by_filename.items():
                if fname_low and fname_low in src_url.lower():
                    matched_ids.add(fdoc["document_id"])

        matched_names: List[str] = []
        link_entry = {
            "chat_id": chat_id,
            "thread_id": thread_id,
            "query": query,
            "answer_summary": answer_summary,
            "created_at": created_at,
        }
        for fid in matched_ids:
            fdoc = by_doc_id.get(fid)
            if not fdoc:
                continue
            matched_names.append(fdoc.get("filename", fid))
            existing_links = [l for l in (fdoc.get("qa_links") or []) if l.get("chat_id") != chat_id]
            updated_links = [link_entry] + existing_links
            self.files.update_one(
                {"document_id": fid},
                {"$set": {"qa_links": updated_links[:50], "qa_count": len(updated_links), "last_queried_at": created_at}},
            )
        return matched_names

    # ── Chat & Q&A Persistence ─────────────────────────────────────────────

    def record_chat_turn(
        self,
        *,
        chat_id: str,
        thread_id: Optional[str],
        query: str,
        report: str,
        sources: List[Dict[str, Any]],
        mode: str,
        rag_mode: str,
        orchestrator: str,
        model: Optional[str],
        answer_origin: str,
        attachment_ids: Optional[List[str]] = None,
        confidence_score: float = 0.85,
        response_accuracy_score: Optional[float] = 0.85,
        processing_time_seconds: float = 0.0,
        memory_context_used: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        resolved_thread = (thread_id or "").strip() or str(uuid.uuid4())
        summary = _summarize_answer(report)
        matched_filenames = self._link_files_to_qa(
            chat_id=chat_id,
            thread_id=resolved_thread,
            query=query,
            answer_summary=summary,
            created_at=now,
            attachment_ids=attachment_ids or [],
            sources=sources or [],
        )
        clean_sources = [
            {
                "citation_id": int(s.get("citation_id", idx + 1)),
                "title": s.get("title", "Source"),
                "url_or_path": s.get("url_or_path", s.get("source", "")),
                "relevance_score": float(s.get("relevance_score", 0.0)),
                "source_type": s.get("source_type", "document"),
                "snippet": (s.get("snippet") or "")[:600],
            }
            for idx, s in enumerate((sources or [])[:15])
        ]
        turn_doc = {
            "chat_id": chat_id,
            "thread_id": resolved_thread,
            "user_id": self.user_id,
            "query": query,
            "report": report,
            "report_summary": summary,
            "sources": clean_sources,
            "attachment_ids": list(attachment_ids or []),
            "attachment_names": matched_filenames,
            "mode": mode,
            "rag_mode": rag_mode,
            "orchestrator": orchestrator,
            "model": model,
            "answer_origin": answer_origin,
            "confidence_score": float(confidence_score or 0.0),
            "response_accuracy_score": float(response_accuracy_score) if response_accuracy_score is not None else None,
            "processing_time_seconds": float(processing_time_seconds or 0.0),
            "memory_context_used": memory_context_used or {"short_term_count": 0, "long_term_count": 0},
            "created_at": now,
        }
        self.chats.update_one({"chat_id": chat_id}, {"$set": turn_doc}, upsert=True)
        return dict(turn_doc)

    def list_chats(
        self,
        *,
        thread_id: Optional[str] = None,
        document_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        filt: Dict[str, Any] = {}
        if thread_id:
            filt["thread_id"] = thread_id
        if document_id:
            fdoc = self.get_file(document_id)
            fname = fdoc.get("filename") if fdoc else None
            or_clauses: List[Dict[str, Any]] = [{"attachment_ids": document_id}]
            if fname:
                or_clauses.append({"attachment_names": fname})
            filt["$or"] = or_clauses
        rows = list(self.chats.find(filt).sort("created_at", -1))
        if search and search.strip():
            needle = search.strip().lower()
            rows = [
                r
                for r in rows
                if needle in (r.get("query") or "").lower()
                or needle in (r.get("report_summary") or "").lower()
                or any(needle in str(fn).lower() for fn in (r.get("attachment_names") or []))
            ]
        rows = rows[: max(1, min(limit, 200))]
        for r in rows:
            r.pop("_id", None)
        return rows

    def list_threads(self, *, search: Optional[str] = None, limit: int = 40) -> List[Dict[str, Any]]:
        all_rows = list(self.chats.find({}).sort("created_at", 1))
        grouped: Dict[str, Dict[str, Any]] = {}
        for r in all_rows:
            r.pop("_id", None)
            tid = r.get("thread_id") or r.get("chat_id")
            if tid not in grouped:
                grouped[tid] = {
                    "thread_id": tid,
                    "title": r.get("query", "Untitled chat"),
                    "latest_chat_id": r.get("chat_id"),
                    "latest_query": r.get("query", ""),
                    "latest_summary": r.get("report_summary", ""),
                    "turn_count": 0,
                    "mode": r.get("mode", "fast"),
                    "model": r.get("model"),
                    "file_names": [],
                    "created_at": r.get("created_at"),
                    "updated_at": r.get("created_at"),
                    "turns": [],
                }
            g = grouped[tid]
            g["turn_count"] += 1
            g["latest_chat_id"] = r.get("chat_id")
            g["latest_query"] = r.get("query", "")
            g["latest_summary"] = r.get("report_summary", "")
            g["mode"] = r.get("mode", g["mode"])
            g["model"] = r.get("model") or g["model"]
            g["updated_at"] = r.get("created_at")
            for fn in r.get("attachment_names") or []:
                if fn not in g["file_names"]:
                    g["file_names"].append(fn)
            g["turns"].append({
                "chat_id": r.get("chat_id"),
                "query": r.get("query"),
                "report": r.get("report"),
                "report_summary": r.get("report_summary"),
                "sources": r.get("sources", []),
                "mode": r.get("mode"),
                "rag_mode": r.get("rag_mode"),
                "orchestrator": r.get("orchestrator"),
                "model": r.get("model"),
                "answer_origin": r.get("answer_origin"),
                "attachment_names": r.get("attachment_names", []),
                "confidence_score": r.get("confidence_score"),
                "response_accuracy_score": r.get("response_accuracy_score"),
                "processing_time_seconds": r.get("processing_time_seconds"),
                "memory_context_used": r.get("memory_context_used"),
                "created_at": r.get("created_at"),
            })
        threads = list(grouped.values())
        if search and search.strip():
            needle = search.strip().lower()
            threads = [
                t
                for t in threads
                if needle in (t.get("title") or "").lower()
                or needle in (t.get("latest_summary") or "").lower()
                or any(needle in (turn.get("query") or "").lower() for turn in t.get("turns", []))
            ]
        threads.sort(key=lambda t: t.get("updated_at") or "", reverse=True)
        return threads[: max(1, min(limit, 100))]

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        doc = self.chats.find_one({"chat_id": chat_id})
        if doc:
            doc.pop("_id", None)
        return doc

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        threads = [t for t in self.list_threads(limit=200) if t["thread_id"] == thread_id]
        return threads[0] if threads else None

    def delete_chat(self, chat_id: str) -> bool:
        res = self.chats.delete_one({"chat_id": chat_id})
        return res.deleted_count > 0

    def delete_thread(self, thread_id: str) -> int:
        res = self.chats.delete_many({"thread_id": thread_id})
        return int(res.deleted_count)

    def get_workspace_stats(self) -> Dict[str, Any]:
        all_chats = list(self.chats.find({}))
        thread_ids = {c.get("thread_id") or c.get("chat_id") for c in all_chats}
        active_files = self.files.count_documents({"deleted": {"$ne": True}})
        stm = self.db["short_term_memory"].count_documents({})
        ltm = self.db["long_term_memory"].count_documents({})
        vs_chunks = 0
        try:
            vs_chunks = get_user_vector_store(self.db_name).get_document_count()
        except Exception:
            pass
        return {
            "db_name": self.db_name,
            "backend": self.manager.backend,
            "total_chats": len(all_chats),
            "total_threads": len(thread_ids),
            "uploaded_files": active_files,
            "indexed_chunks": vs_chunks,
            "short_term_memories": stm,
            "long_term_memories": ltm,
        }
