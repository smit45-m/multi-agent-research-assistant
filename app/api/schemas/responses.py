from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class SourceInfo(BaseModel):
    """Information about a retrieved source."""
    title: str
    url_or_path: str
    relevance_score: float
    source_type: str

class ResearchResponse(BaseModel):
    """Response model for a completed or ongoing research task."""
    task_id: str
    status: str
    query: str
    report: Optional[str] = None
    sources: List[SourceInfo] = []
    confidence_score: float = 0.0
    processing_time_seconds: float = 0.0
    created_at: datetime

class DocumentResponse(BaseModel):
    """Response model for a single document."""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    uploaded_at: datetime

class DocumentListResponse(BaseModel):
    """Response model for a list of documents."""
    documents: List[DocumentResponse]
    total_count: int

class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    status: str
    version: str
    uptime_seconds: float
    vector_store_documents: int

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int
