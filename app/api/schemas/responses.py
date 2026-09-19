"""
Pydantic v2 response models for schema validation.

Metric fields default to 0.0 — they are only populated with values that
were actually measured by the pipeline.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """Information about a retrieved source."""

    title: str = Field(description="Title or filename of the source.")
    url_or_path: str = Field(description="URL or filesystem path.")
    relevance_score: float = Field(
        description="Retrieval relevance score (0.0 to 1.0)."
    )
    source_type: str = Field(
        description="Source category: 'pdf', 'arxiv', 'web', 'csv', etc."
    )
    snippet: Optional[str] = Field(
        default=None, description="Extracted snippet or excerpt."
    )


class AgentTelemetryInfo(BaseModel):
    """Per-agent measured execution latency breakdown."""

    planner_time_ms: float = 0.0
    retriever_time_ms: float = 0.0
    analyzer_time_ms: float = 0.0
    writer_time_ms: float = 0.0
    verifier_time_ms: float = 0.0
    critic_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    sequential_retrieval_baseline_ms: float = 0.0
    parallel_retrieval_actual_ms: float = 0.0
    synthesis_reduction_pct: float = 0.0


class VerificationInfo(BaseModel):
    """Verifier agent output: measured grounding of the report."""

    grounded_ratio: float = 0.0
    total_sentences: int = 0
    grounded_sentences: int = 0
    citation_integrity: float = 0.0
    measured_accuracy: float = 0.0
    method: str = ""
    ungrounded_sentences: List[str] = Field(default_factory=list)


class CritiqueInfo(BaseModel):
    """Critic agent output: measured quality review."""

    quality_score: float = 0.0
    grounding_component: float = 0.0
    structure_component: float = 0.0
    evidence_component: float = 0.0
    revision_notes: List[str] = Field(default_factory=list)
    needs_revision: bool = False
    method: str = ""


class ResearchResponse(BaseModel):
    """Response model for a completed or ongoing research task."""

    task_id: str
    status: str = Field(description="Task status: 'pending', 'completed', 'failed'")
    query: str
    report: Optional[str] = Field(
        default=None, description="Generated research report in Markdown."
    )
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources cited in the report."
    )
    confidence_score: float = Field(
        default=0.0, description="Measured retrieval/analysis confidence."
    )
    response_accuracy_score: float = Field(
        default=0.0,
        description="Verifier-measured grounding accuracy (0.0 to 1.0).",
    )
    synthesis_speedup_ratio: float = Field(
        default=0.0,
        description="Measured parallel retrieval speedup ratio (0.0 to 1.0).",
    )
    processing_time_seconds: float = Field(
        default=0.0, description="Measured end-to-end execution time."
    )
    orchestrator: str = Field(
        default="langgraph", description="Framework: 'langgraph' or 'crewai'."
    )
    telemetry: Optional[AgentTelemetryInfo] = None
    verification: Optional[VerificationInfo] = None
    critique: Optional[CritiqueInfo] = None
    revision_count: int = 0
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
    llm_configured: bool = False
    web_search_available: bool = False


class BenchmarkResponse(BaseModel):
    """Response model for the measured benchmark suite."""

    total_test_cases: int
    passed_test_cases: int
    pass_rate_percentage: float
    average_accuracy_percentage: float
    target_accuracy_percentage: float = 85.0
    average_latency_seconds: float
    p95_latency_seconds: float = 0.0
    target_latency_seconds: float = 8.0
    average_synthesis_speedup_percentage: float
    target_synthesis_speedup_percentage: float = 60.0
    categories_evaluated: List[str]
    llm_mode: str = "extractive-fallback"
    measurement_note: str = ""
    sample_results: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    detail: Optional[str] = None
    status_code: int
