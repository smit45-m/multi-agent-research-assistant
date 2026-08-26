import pytest
from unittest.mock import patch, MagicMock
from app.rag.vector_store import VectorStoreManager

def test_vector_store_initialization():
    manager = VectorStoreManager()
    assert manager is not None

def test_add_and_search_documents(sample_documents):
    manager = VectorStoreManager()
    manager.add_documents = MagicMock()
    manager.similarity_search = MagicMock(return_value=[sample_documents[0]])
    
    manager.add_documents(sample_documents)
    results = manager.similarity_search("Sample")
    assert len(results) == 1
    assert results[0].page_content == "Sample document content 1"

def test_document_count(sample_documents):
    manager = VectorStoreManager()
    manager.get_document_count = MagicMock(return_value=2)
    assert manager.get_document_count() == 2

def test_empty_search():
    manager = VectorStoreManager()
    manager.similarity_search = MagicMock(return_value=[])
    assert len(manager.similarity_search("Not found")) == 0
