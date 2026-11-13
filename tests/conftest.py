import pytest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*renamed to.*")

import math
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

@pytest.fixture(autouse=True)
def fake_atlas_vector_search(monkeypatch):
    """Unit tests never reach Atlas: simulate `$vectorSearch` with brute-force cosine over the mongomock collection.

    The real `$vectorSearch` pipeline is covered by `test_atlas_vector_search_pipeline_is_owner_filtered`
    and by the live post-deploy verification against the official Atlas cluster.
    """
    from app.rag.atlas_vector_store import AtlasVectorStore

    def _noop_index(self):
        return None

    def _brute_force_search(self, query_vector, k):
        qn = math.sqrt(sum(x * x for x in query_vector)) or 1.0
        scored = []
        for row in self.vectors.find({"owner_db": self.db_name}):
            vec = row["embedding"]
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            cosine = sum(a * b for a, b in zip(query_vector, vec)) / (qn * vn)
            scored.append((str(row["_id"]), (1.0 + cosine) / 2.0))  # Atlas cosine score
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    monkeypatch.setattr(AtlasVectorStore, "_ensure_vector_index", _noop_index)
    monkeypatch.setattr(AtlasVectorStore, "_vector_search", _brute_force_search)


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
         patch("app.rag.atlas_vector_store.get_settings", return_value=mock_settings), \
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
