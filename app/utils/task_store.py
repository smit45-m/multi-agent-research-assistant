"""
Task result store with pluggable backends.

- RedisTaskStore: shared state for multi-replica deployments (REDIS_URL set).
- InMemoryTaskStore: single-process fallback with TTL-based eviction.

Both stores serialize ResearchResponse models as JSON.
"""

import threading
import time
from typing import Optional, Protocol

from app.api.schemas.responses import ResearchResponse
from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_TASK_TTL_S = 3600
_MAX_IN_MEMORY_TASKS = 1000


class TaskStore(Protocol):
    """Interface shared by all task store backends."""

    def set(self, task_id: str, response: ResearchResponse) -> None: ...

    def get(self, task_id: str) -> Optional[ResearchResponse]: ...


class InMemoryTaskStore:
    """Thread-safe in-process task store with TTL eviction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, tuple[float, ResearchResponse]] = {}

    def set(self, task_id: str, response: ResearchResponse) -> None:
        with self._lock:
            self._tasks[task_id] = (time.monotonic(), response)
            self._evict_locked()

    def get(self, task_id: str) -> Optional[ResearchResponse]:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            created, response = entry
            if time.monotonic() - created > _TASK_TTL_S:
                del self._tasks[task_id]
                return None
            return response

    def _evict_locked(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._tasks.items() if now - ts > _TASK_TTL_S]
        for k in expired:
            del self._tasks[k]
        while len(self._tasks) > _MAX_IN_MEMORY_TASKS:
            oldest = min(self._tasks, key=lambda k: self._tasks[k][0])
            del self._tasks[oldest]


class RedisTaskStore:
    """Redis-backed task store for horizontally scaled deployments."""

    def __init__(self, redis_url: str) -> None:
        import redis  # optional dependency, validated by factory

        self._client = redis.Redis.from_url(
            redis_url, socket_timeout=3, decode_responses=True
        )
        self._client.ping()

    @staticmethod
    def _key(task_id: str) -> str:
        return f"research:task:{task_id}"

    def set(self, task_id: str, response: ResearchResponse) -> None:
        self._client.setex(self._key(task_id), _TASK_TTL_S, response.model_dump_json())

    def get(self, task_id: str) -> Optional[ResearchResponse]:
        raw = self._client.get(self._key(task_id))
        if raw is None:
            return None
        return ResearchResponse.model_validate_json(str(raw))


def create_task_store() -> "TaskStore":
    """
    Build the appropriate task store for this deployment.

    Uses Redis when REDIS_URL is configured (required for replicas > 1);
    otherwise an in-process store.
    """
    settings = get_settings()
    if settings.REDIS_URL:
        try:
            store = RedisTaskStore(settings.REDIS_URL)
            logger.info("Using Redis task store at %s", settings.REDIS_URL)
            return store
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            logger.warning(
                "Redis unavailable (%s); falling back to in-memory task "
                "store. Do NOT run multiple replicas in this mode.",
                exc,
            )
    return InMemoryTaskStore()
