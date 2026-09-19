"""
Health and readiness probes for Kubernetes, Docker, and monitoring systems.
"""

import time

from fastapi import APIRouter, Request

from app.api.schemas.responses import HealthResponse
from app.config import get_settings

router = APIRouter(prefix="/health", tags=["Health"])
START_TIME = time.time()


def _build_health(request: Request, check_readiness: bool) -> HealthResponse:
    """Shared health payload builder."""
    settings = get_settings()
    uptime = time.time() - START_TIME
    vector_store = getattr(request.app.state, "vector_store", None)

    doc_count = 0
    status = "ok"
    if vector_store:
        try:
            doc_count = vector_store.get_document_count()
        except Exception:  # noqa: BLE001 - degraded, not dead
            if check_readiness:
                status = "degraded"
    elif check_readiness:
        status = "not_initialized"

    web_available = False
    try:
        from app.tools.search_tool import WebSearchTool

        web_available = WebSearchTool().is_available()
    except Exception:  # noqa: BLE001
        pass

    return HealthResponse(
        status=status,
        version="2.0.0",
        uptime_seconds=round(uptime, 1),
        vector_store_documents=doc_count,
        llm_configured=settings.llm_available,
        web_search_available=web_available,
    )


@router.get("", response_model=HealthResponse, include_in_schema=False)
@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic liveness probe."""
    return _build_health(request, check_readiness=False)


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(request: Request) -> HealthResponse:
    """Readiness probe that checks vector store state."""
    return _build_health(request, check_readiness=True)
