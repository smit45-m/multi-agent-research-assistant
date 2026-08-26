"""
FAISS vector store management for document retrieval.
"""
import os
import threading
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.config import get_settings
from app.utils.logger import setup_logger
from app.rag.embeddings import get_embedding_model
from app.utils.exceptions import RetrievalError

logger = setup_logger(__name__, "INFO")

class VectorStoreManager:
    """
    Thread-safe manager for a FAISS vector store.
    """
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initializes the VectorStoreManager.
        """
        settings = get_settings()
        self.persist_directory = persist_directory or settings.VECTOR_STORE_PATH
        self.embeddings_model = get_embedding_model()
        self.vector_store: Optional[FAISS] = None
        self._lock = threading.Lock()
        
    def _load_or_create_store(self) -> Optional[FAISS]:
        """
        Internal method to load an existing FAISS index from disk.
        """
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                logger.info(f"Loading existing FAISS index from {self.persist_directory}...")
                store = FAISS.load_local(
                    folder_path=self.persist_directory,
                    embeddings=self.embeddings_model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded existing FAISS index.")
                return store
            else:
                logger.info("No existing index found. Returning None to indicate empty store.")
                return None
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise RetrievalError(f"Failed to load vector store: {str(e)}")

    def initialize(self) -> None:
        """
        Initializes the vector store by loading it from disk.
        """
        with self._lock:
            if self.vector_store is None:
                self.vector_store = self._load_or_create_store()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adds a list of documents to the vector store.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return []
            
        logger.info(f"Adding {len(documents)} documents to the vector store...")
        with self._lock:
            try:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(
                        documents,
                        self.embeddings_model
                    )
                    logger.info("Created new FAISS index with provided documents.")
                    doc_ids = []
                else:
                    doc_ids = self.vector_store.add_documents(documents)
                    logger.info(f"Added documents to existing FAISS index.")
                    return doc_ids if doc_ids else []
                return doc_ids
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}")
                raise RetrievalError(f"Failed to add documents: {str(e)}")

    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Document]:
        """
        Performs a similarity search on the vector store.
        """
        logger.info(f"Performing similarity search for query: '{query}' (k={k})")
        with self._lock:
            if self.vector_store is None:
                logger.warning("Vector store is empty or uninitialized. Returning no results.")
                return []
                
            try:
                results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
                filtered_docs = []
                for doc, score in results:
                    if score >= score_threshold:
                        doc.metadata["relevance_score"] = score
                        filtered_docs.append(doc)
                        
                logger.info(f"Found {len(filtered_docs)} relevant documents (above threshold {score_threshold}).")
                return filtered_docs
            except NotImplementedError:
                logger.warning("Relevance scores not supported by current FAISS index configuration. Falling back to standard search.")
                docs = self.vector_store.similarity_search(query, k=k)
                return docs
            except Exception as e:
                logger.error(f"Error during similarity search: {str(e)}")
                raise RetrievalError(f"Search failed: {str(e)}")

    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Deletes documents from the vector store by their IDs.
        """
        logger.info(f"Deleting {len(doc_ids)} documents from vector store...")
        with self._lock:
            if self.vector_store is None:
                logger.warning("Cannot delete from an empty/uninitialized vector store.")
                return
                
            try:
                self.vector_store.delete(doc_ids)
                logger.info("Successfully deleted documents.")
            except Exception as e:
                logger.error(f"Error deleting documents: {str(e)}")
                raise RetrievalError(f"Failed to delete documents: {str(e)}")

    def get_document_count(self) -> int:
        """
        Returns the total number of documents (vectors) in the store.
        """
        with self._lock:
            if self.vector_store is None:
                return 0
            
            try:
                count = self.vector_store.index.ntotal
                return count
            except Exception as e:
                logger.error(f"Error getting document count: {str(e)}")
                return 0

    def save(self) -> None:
        """
        Persists the current state of the vector store to disk.
        """
        with self._lock:
            if self.vector_store is None:
                logger.warning("No vector store to save.")
                return
                
            logger.info(f"Saving FAISS index to {self.persist_directory}...")
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.vector_store.save_local(self.persist_directory)
                logger.info("Successfully saved FAISS index to disk.")
            except Exception as e:
                logger.error(f"Error saving vector store: {str(e)}")
                raise RetrievalError(f"Failed to save vector store: {str(e)}")
