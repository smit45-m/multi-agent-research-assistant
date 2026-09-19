"""
Pytest test configuration and mock fixtures.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from starlette.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.initialize = MagicMock()
    store.add_documents = MagicMock(return_value=["doc-1"])
    store.similarity_search = MagicMock(
        return_value=[
            Document(
                page_content="Sample document content 1",
                metadata={"source": "test1.txt", "relevance_score": 0.9},
            )
        ]
    )
    store.get_document_count = MagicMock(return_value=2)
    store.save = MagicMock()
    return store


@pytest.fixture
def test_client(mock_vector_store):
    """
    TestClient wired with a mock vector store.

    Sends the configured API key header automatically (when API_KEY is set
    in the environment) so the suite passes regardless of auth settings.
    """
    app = create_app()
    app.state.vector_store = mock_vector_store
    client = TestClient(app)

    settings = get_settings()
    if settings.API_KEY:
        client.headers.update({"X-API-Key": settings.API_KEY})
    yield client


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Sample document content 1",
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="Sample document content 2",
            metadata={"source": "test2.txt"},
        ),
    ]


@pytest.fixture
def mock_llm_response():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"key_findings": ["Factual finding"], "confidence_score": 0.9}'
    )
    return mock_llm
