import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.config import Settings
from langchain_core.documents import Document

@pytest.fixture
def mock_settings():
    return Settings(
        openai_api_key="sk-test-key",
        environment="test",
        debug=True,
        log_level="DEBUG"
    )

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.initialize = MagicMock()
    store.add_documents = MagicMock()
    store.similarity_search = MagicMock(return_value=[])
    store.get_document_count = MagicMock(return_value=0)
    store.save = MagicMock()
    return store

@pytest.fixture
def test_client(mock_settings, mock_vector_store):
    with patch("app.config.get_settings", return_value=mock_settings):
        with patch("app.main.vector_store", mock_vector_store):
            app = create_app()
            transport = ASGITransport(app=app)
            client = AsyncClient(transport=transport, base_url="http://test")
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
    mock_llm.invoke.return_value.content = "Mocked LLM Response"
    return mock_llm
