"""
Web search tool for agents using DuckDuckGo (ddgs).

Includes a cached availability probe so that offline environments fail fast
once instead of burning seconds per request on doomed network calls, plus a
small in-process TTL result cache to deduplicate identical queries under
concurrent load.
"""

import threading
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_CACHE_TTL_S = 300.0
_AVAILABILITY_RECHECK_S = 60.0


def _import_ddgs() -> Any:
    """Import the DDGS client from either the new or legacy package."""
    try:
        from ddgs import DDGS as ModernDDGS  # modern package name

        return ModernDDGS
    except ImportError:
        from duckduckgo_search import DDGS as LegacyDDGS  # legacy fallback

        return LegacyDDGS


class WebSearchTool:
    """Wrapper for DuckDuckGo search with availability probing and caching."""

    _lock = threading.Lock()
    _available: Optional[bool] = None
    _last_check: float = 0.0
    _cache: Dict[str, tuple] = {}

    def is_available(self) -> bool:
        """
        Check (and cache) whether outbound web search is usable.
        Rechecks periodically so transient outages recover.
        """
        settings = get_settings()
        if not settings.WEB_SEARCH_ENABLED:
            return False

        cls = type(self)
        now = time.monotonic()
        with cls._lock:
            if (
                cls._available is not None
                and now - cls._last_check < _AVAILABILITY_RECHECK_S
            ):
                return cls._available

        ok = self._probe()
        with cls._lock:
            cls._available = ok
            cls._last_check = time.monotonic()
        return ok

    @staticmethod
    def _probe() -> bool:
        """Cheap TCP-level connectivity probe (no full search)."""
        import socket

        try:
            with socket.create_connection(("duckduckgo.com", 443), timeout=2.0):
                return True
        except OSError:
            logger.info(
                "Web search unavailable (no outbound network); "
                "external retrieval will be skipped."
            )
            return False

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search using DuckDuckGo.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            A list of dicts containing title, url, and snippet.
        """
        if not self.is_available():
            return []

        cls = type(self)
        cache_key = f"{query}::{max_results}"
        now = time.monotonic()
        with cls._lock:
            cached = cls._cache.get(cache_key)
            if cached and now - cached[0] < _CACHE_TTL_S:
                return list(cached[1])

        try:
            ddgs_cls = _import_ddgs()
            with ddgs_cls() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
            with cls._lock:
                cls._cache[cache_key] = (time.monotonic(), formatted)
                if len(cls._cache) > 256:
                    oldest = min(cls._cache, key=lambda k: cls._cache[k][0])
                    del cls._cache[oldest]
            return formatted
        except Exception as exc:  # noqa: BLE001 - search is best-effort
            logger.warning("Error during web search: %s", exc)
            return []


@tool
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a web search to find relevant information.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.
    """
    searcher = WebSearchTool()
    return searcher.search(query, max_results)
