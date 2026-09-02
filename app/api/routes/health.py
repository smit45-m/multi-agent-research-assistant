"""
Health and readiness probes for Kubernetes, Docker, and monitoring systems.
"""
import time
from fastapi import APIRouter, Request
from app.api.schemas.responses import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])
START_TIME = time.time()

@router.get("", response_model=HealthResponse, include_in_schema=False)
@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic liveness probe."""
    uptime = time.time() - START_TIME
    vector_store = getattr(request.app.state, "vector_store", None)
    doc_count = 0
    if vector_store:
        try:
            doc_count = vector_store.get_document_count()
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(uptime, 1),
        vector_store_documents=doc_count,
        active_concurrent_capacity=50,
        latency_sla_seconds=8.0,
        accuracy_benchmark_target=85.0
    )

@router.get("/ready", response_model=HealthResponse)
async def readiness_check(request: Request):
    """Readiness probe that checks vector store state."""
    uptime = time.time() - START_TIME
    vector_store = getattr(request.app.state, "vector_store", None)
    
    doc_count = 0
    status = "ok"
    if vector_store:
        try:
            doc_count = vector_store.get_document_count()
        except Exception:
            status = "degraded"
    else:
        status = "not_initialized"
        
    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=round(uptime, 1),
        vector_store_documents=doc_count,
        active_concurrent_capacity=50,
        latency_sla_seconds=8.0,
        accuracy_benchmark_target=85.0
    )
