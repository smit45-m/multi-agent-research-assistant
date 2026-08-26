import time
from fastapi import APIRouter, Request
from app.api.schemas.responses import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic liveness probe."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=uptime,
        vector_store_documents=0
    )

@router.get("/ready", response_model=HealthResponse)
async def readiness_check(request: Request):
    """Readiness probe that checks dependencies like vector store."""
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
        uptime_seconds=uptime,
        vector_store_documents=doc_count
    )
