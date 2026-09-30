"""Thread-safe canonical document store with optional normalized FAISS indexing.

JSON, not executable pickle, persists documents. Lexical retrieval works even when
semantic model loading fails. Existing legacy index files are never removed.
"""
import json
import os
import threading
import uuid
from pathlib import Path
from langchain_core.documents import Document
from app.config import get_settings
from app.rag.embeddings import get_embedding_model
from app.utils.exceptions import RetrievalError


class VectorStoreManager:
    def __init__(self, persist_directory=None, embeddings=None):
        self.persist_directory = str(persist_directory or get_settings().store_path)
        self.embeddings_model = embeddings
        self.vector_store = None
        self._documents = {}
        self._lock = threading.RLock()
        self.version = 0
        self.initialized = False
        self.embedding_status = "not_loaded"
        self.warnings = []
        self._embedding_attempted = embeddings is not None

    @property
    def dense_available(self):
        return self.vector_store is not None and self.embedding_status == "ready"

    def initialize(self):
        with self._lock:
            if self.initialized:
                return
            path = Path(self.persist_directory) / "documents.json"
            if path.is_file():
                try:
                    items = json.loads(path.read_text(encoding="utf-8"))
                    self._documents = {item["id"]: Document(page_content=item["content"], metadata=item["metadata"]) for item in items}
                except (ValueError, KeyError, TypeError) as exc:
                    raise RetrievalError("Could not read the persisted JSON document index.") from exc
            elif (Path(self.persist_directory) / "index.pkl").exists():
                self.warnings.append("Legacy pickle index was not deserialized. Re-upload original documents to migrate safely; legacy files remain untouched.")
            self.initialized = True
            self.version += 1

    def prepare_dense(self):
        """Call at startup or explicit ingestion, never download models during a query."""
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
            self._rebuild_dense()
            return self.embedding_status == "ready"

    def _rebuild_dense(self):
        if self.embeddings_model is None or not self._documents:
            self.vector_store = None
            return
        try:
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.from_documents(
                list(self._documents.values()), self.embeddings_model,
                ids=list(self._documents), normalize_L2=True)
            self.embedding_status = "ready"
        except Exception:
            self.vector_store = None
            self.embedding_status = "unavailable"
            warning = "Semantic indexing failed; uploaded text is preserved and keyword retrieval remains available."
            if warning not in self.warnings:
                self.warnings.append(warning)

    def add_documents(self, documents):
        if not documents:
            return []
        with self._lock:
            ids = []
            for doc in documents:
                if not doc.page_content.strip():
                    continue
                identifier = str(uuid.uuid4())
                self._documents[identifier] = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                ids.append(identifier)
            self.version += 1
            if self.embeddings_model is not None:
                self._rebuild_dense()
            return ids

    def get_all_documents(self):
        with self._lock:
            return [Document(page_content=d.page_content, metadata={**d.metadata, "chunk_id": key}) for key, d in self._documents.items()]

    def similarity_search(self, query, k=5, score_threshold=0.25):
        if not query.strip() or k < 1:
            return []
        with self._lock:
            if not self.dense_available:
                return []
            try:
                # Normalized vectors: squared L2 = 2 - 2*cosine similarity.
                matches = self.vector_store.similarity_search_with_score(query, k=k)
                result = []
                for doc, distance in matches:
                    score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
                    if score >= score_threshold:
                        result.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "dense_score": score, "relevance_score": score}))
                return result
            except Exception:
                raise RetrievalError("Semantic search failed; keyword retrieval remains available.") from None

    def delete_documents(self, doc_ids):
        with self._lock:
            for identifier in doc_ids:
                self._documents.pop(identifier, None)
            self.version += 1
            if self.embeddings_model is not None:
                self._rebuild_dense()

    def delete_by_document_id(self, document_id):
        with self._lock:
            ids = [key for key, doc in self._documents.items() if doc.metadata.get("document_id") == document_id]
            self.delete_documents(ids)
            return len(ids)

    def get_document_count(self):
        with self._lock:
            return len(self._documents)

    def save(self):
        with self._lock:
            directory = Path(self.persist_directory)
            if not self._documents and not (directory / "documents.json").exists():
                return
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "documents.json"
            temporary = directory / ("documents." + uuid.uuid4().hex + ".tmp")
            payload = [{"id": identifier, "content": doc.page_content, "metadata": doc.metadata} for identifier, doc in self._documents.items()]
            temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(temporary, target)
