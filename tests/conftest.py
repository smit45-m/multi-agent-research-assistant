"""
Pytest test configuration and mock fixtures.
"""
import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from app.main import create_app
from app.config import Settings
from langchain_core.documents import Document

@pytest.fixture
def mock_settings():
    return Settings(
        OPENAI_API_KEY="sk-test-key",
        LOG_LEVEL="INFO"
    )

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.initialize = MagicMock()
    store.add_documents = MagicMock(return_value=["doc-1"])
    store.similarity_search = MagicMock(return_value=[
        Document(page_content="Sample document content 1", metadata={"source": "test1.txt", "relevance_score": 0.9})
    ])
    store.get_document_count = MagicMock(return_value=2)
    store.save = MagicMock()
    return store

@pytest.fixture
def test_client(mock_settings, mock_vector_store):
    with patch("app.config.get_settings", return_value=mock_settings):
        with patch("app.main.vector_store", mock_vector_store):
            app = create_app()
            app.state.vector_store = mock_vector_store
            client = TestClient(app)
            yield client

@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Sample document content 1", metadata={"source": "test1.txt"}),
        Document(page_content="Sample document content 2", metadata={"source": "test2.txt"}),
    ]

@pytest.fixture
def mock_llm_response():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"key_findings": ["Factual finding"], "confidence_score": 0.9}'
    return mock_llm
