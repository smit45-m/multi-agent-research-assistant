"""
Tests for the pluggable task store.
"""

from app.api.schemas.responses import ResearchResponse
from app.utils.task_store import InMemoryTaskStore


def _response(task_id: str) -> ResearchResponse:
    return ResearchResponse(task_id=task_id, status="completed", query="q")


def test_in_memory_store_set_get():
    store = InMemoryTaskStore()
    store.set("abc", _response("abc"))
    got = store.get("abc")
    assert got is not None
    assert got.task_id == "abc"


def test_in_memory_store_missing_returns_none():
    store = InMemoryTaskStore()
    assert store.get("missing") is None


def test_in_memory_store_eviction_caps_size():
    store = InMemoryTaskStore()
    for i in range(1100):
        store.set(f"task-{i}", _response(f"task-{i}"))
    # Oldest entries evicted; newest retained.
    assert store.get("task-0") is None
    assert store.get("task-1099") is not None
