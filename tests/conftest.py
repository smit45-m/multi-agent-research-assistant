import pytest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*renamed to.*")

from unittest.mock import MagicMock, patch

import mongomock
from starlette.testclient import TestClient

from app.main import create_app
from app.config import Settings
from langchain_core.documents import Document

@pytest.fixture(autouse=True)
def mock_search_tool():
    with patch("app.tools.search_tool.WebSearchTool.search") as mock_s:
        mock_s.return_value = [
            {"title": "Autonomous Agents Overview", "url": "https://example.com/agents", "snippet": "Autonomous agents are AI systems capable of perceiving and acting in their environment.", "source_type": "web"}
        ]
        yield mock_s

@pytest.fixture
def mock_settings(tmp_path):
    return Settings(
        OPENAI_API_KEY="sk-test-key",
        LOG_LEVEL="INFO",
        MONGODB_URI="",
        USER_WORKSPACES_DIR=str(tmp_path / "user_workspaces"),
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
def test_client(mock_settings, mock_vector_store, monkeypatch):
    from app.db.mongo_engine import reset_mongo_manager
    from app.db.user_repository import clear_cached_user_vector_stores
    monkeypatch.setenv("MONGODB_URI", "")
    reset_mongo_manager(client=mongomock.MongoClient())
    clear_cached_user_vector_stores()
    with patch("app.config.get_settings", return_value=mock_settings), \
         patch("app.main.get_settings", return_value=mock_settings), \
         patch("app.db.mongo_engine.get_settings", return_value=mock_settings), \
         patch("app.db.user_repository.get_settings", return_value=mock_settings), \
         patch("app.main.vector_store", mock_vector_store):
        app = create_app()
        app.state.vector_store = mock_vector_store
        with TestClient(app) as client:
            yield client
    clear_cached_user_vector_stores()

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
